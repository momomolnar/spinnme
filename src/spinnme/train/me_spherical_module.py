import copy
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from astropy import units as u
from astropy.visualization import ImageNormalize
from matplotlib.colors import Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from spinnme.instrument import resolve_instrument_line
from spinnme.model import (
    LimbCorrectionModel,
    MESphericalModel,
    MESphericalSirenModel,
    NormalizationModule,
    VelocityCorrectionModel,
)
from spinnme.train.artifact_correction import ArtifactCorrectionModule
from spinnme.train.me_atmosphere import HMIMEAtmosphere, MEAtmosphere, PHIMEAtmosphere
from spinnme.train.metrics import TrainingMetricWindow
from spinnme.train.numerics import write_numerical_report
from spinnme.train.physics import PhysicsWeightSchedule
from spinnme.train.physics_regularization import (
    PhysicsRegularization,
    split_stokes_and_physics_config,
)
from spinnme.train.soap import SOAP
from spinnme.train.spectral_response import InstrumentSpectralResponse
from spinnme.train.spherical_synthesis import (
    correct_spherical_limb_effects,
    field_free_forward_parameters,
    mix_magnetic_filling_factor,
    scale_spherical_forward_parameters,
    transform_spherical_parameters,
)
from spinnme.train.stokes_loss import StokesLossModule
from spinnme.train.util import log_wandb_image, random_time_shift_coords, to_cartesian, to_spherical


class MESphericalModule(LightningModule):

    def __init__(self, image_shape, wavelength_config,
                 gauss_per_dB, Rs_per_ds, seconds_per_dt,
                 instrument_config,
                 lr_params=None, model_config=None, normalization_config=None,
                 stokes_loss_config=None,
                 physics_config=None, physics_domain=None,
                 time_shift_config=None,
                 weights_config=None, optimizer_config=None, instrument_weights=None, wavelength_reduction="sum",
                 replay_config=None, metric_log_interval=50, plot_validation=True,
                 compile_config=None):
        super().__init__()
        for name, scale in (('gauss_per_dB', gauss_per_dB), ('Rs_per_ds', Rs_per_ds),
                            ('seconds_per_dt', seconds_per_dt)):
            if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        self.snapshot_init = copy.deepcopy({
            'image_shape': image_shape, 'wavelength_config': wavelength_config,
            'gauss_per_dB': gauss_per_dB, 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt,
            'instrument_config': instrument_config, 'lr_params': lr_params, 'model_config': model_config,
            'normalization_config': normalization_config, 'stokes_loss_config': stokes_loss_config,
            'physics_config': physics_config, 'physics_domain': physics_domain,
            'time_shift_config': time_shift_config, 'weights_config': weights_config,
            'optimizer_config': optimizer_config, 'instrument_weights': instrument_weights,
            'wavelength_reduction': wavelength_reduction, 'metric_log_interval': metric_log_interval,
            'compile_config': compile_config, 'plot_validation': plot_validation,
        })
        if replay_config is not None:
            raise ValueError('Replay sampling was removed; use opt-in polarization oversampling instead.')
        if wavelength_reduction not in ('sum', 'mean'):
            raise ValueError('wavelength_reduction must be sum or mean.')
        self.wavelength_reduction = wavelength_reduction
        self.instrument_weights = dict(instrument_weights or {})
        if any(not np.isfinite(value) or value <= 0 for value in self.instrument_weights.values()):
            raise ValueError('instrument_weights must be finite and positive.')
        unknown_weights = self.instrument_weights.keys() - {item['instrument_id'] for item in instrument_config}
        if unknown_weights:
            raise ValueError(f'Weights name unknown instruments: {sorted(unknown_weights)}.')
        self.instrument_config = copy.deepcopy(instrument_config)
        self.wavelength_config = copy.deepcopy(wavelength_config)
        self.instrument_lines = {}
        self.line_atomic_parameters = {}
        for item in instrument_config:
            instrument_id = item['instrument_id']
            if instrument_id in self.instrument_lines:
                raise ValueError(f'Duplicate instrument identifier: {instrument_id!r}.')
            if instrument_id not in wavelength_config:
                raise ValueError(f'Missing wavelength configuration for {instrument_id!r}.')
            line, atomic = resolve_instrument_line(item, wavelength_config[instrument_id])
            previous = self.line_atomic_parameters.setdefault(line, atomic)
            if previous != atomic:
                raise ValueError(f'Inconsistent atomic parameters for spectral line {line!r}.')
            self.instrument_lines[instrument_id] = line
        lr_params = lr_params if lr_params is not None else {"start": 1e-3, "end": 1e-4, "iterations": 1e5}

        if isinstance(metric_log_interval, bool) or not isinstance(metric_log_interval, int) or metric_log_interval < 1:
            raise ValueError('metric_log_interval must be a positive integer.')
        if not isinstance(plot_validation, bool):
            raise ValueError('plot_validation must be a boolean.')
        self.plot_validation = plot_validation
        self.metric_log_interval = metric_log_interval
        self._train_metrics = TrainingMetricWindow()
        self.image_shape = image_shape

        # init model
        model_config = copy.deepcopy(model_config) if model_config is not None else {}
        model_type = model_config.pop('type', None)
        model_config['line_ids'] = list(dict.fromkeys(self.instrument_lines.values()))
        model_config['instrument_lines'] = dict(self.instrument_lines)
        self.model_config = {'type': model_type, **copy.deepcopy(model_config)}
        # Configuration scales are physical: gauss for B and km/s for velocity.
        # Parameter models operate in the normalized dataset units.
        if model_config.get('b_sinh_scale') is not None:
            model_config['b_sinh_scale'] /= gauss_per_dB
        if model_config.get('v_sinh_scale') is not None:
            km_per_ds = Rs_per_ds * (1 * u.Rsun).to_value(u.km)
            model_config['v_sinh_scale'] *= seconds_per_dt / km_per_ds
        normalized_profiles = (stokes_loss_config or {}).get('continuum_normalization', False)
        expected_source = 'ratio' if normalized_profiles else 'independent'
        model_config.setdefault('source_function', expected_source)
        self.model_config["source_function"] = model_config["source_function"]
        self.snapshot_init['model_config'] = copy.deepcopy(self.model_config)
        if model_config['source_function'] != expected_source:
            raise ValueError('Continuum-normalized fits require source_function=ratio; '
                             'absolute-intensity fits require source_function=independent.')
        if model_type == 'mlp':
            self.parameter_model = MESphericalModel(**model_config)
        elif model_type == 'siren':
            self.parameter_model = MESphericalSirenModel(**model_config)
        else:
            raise ValueError("The spherical model configuration must specify type: mlp or siren.")

        # init instrument models
        forward_models = {}
        velocity_correction_models = {}
        limb_correction_models = {}
        artifact_correction_models = {}
        hmi_instrument_ids = []
        spectral_responses = {}
        for instrument in copy.deepcopy(instrument_config):
            instrument_id = instrument.pop('instrument_id')
            instrument_type = instrument.pop('type').lower()
            line_id = self.instrument_lines[instrument_id]
            instrument.pop('line_id', None)
            instrument.pop('line', None)
            instrument.pop('atomic_parameters', None)
            instrument.pop('line_config', None)
            for atomic_key in ('j_up', 'j_low', 'g_up', 'g_low'):
                instrument.pop(atomic_key, None)
            response_config = instrument.pop('spectral_response', None)
            if instrument_type == 'hmi':
                if response_config not in (None, 'acquisition_calibration_required'):
                    raise ValueError('HMI requires acquisition-dependent calibration, not a generic passband.')
            else:
                spectral_responses[instrument_id] = InstrumentSpectralResponse(response_config)
            wavelength_center = wavelength_config[instrument_id]['wavelength_center']
            velocity_correction = instrument.pop('velocity_correction', False)
            limb_correction = instrument.pop('limb_correction', False)
            artifact_correction = instrument.pop('artifact_correction', False)

            if instrument_type == 'hmi':
                forward_models[instrument_id] = HMIMEAtmosphere(wavelength_center=wavelength_center, **instrument)
                hmi_instrument_ids.append(instrument_id)
            elif instrument_type == 'phi':
                forward_models[instrument_id] = PHIMEAtmosphere(wavelength_center=wavelength_center, **instrument)
            elif instrument_type in ('generic', 'me', 'hinode'):
                forward_models[instrument_id] = MEAtmosphere(wavelength_center=wavelength_center,
                    **self.line_atomic_parameters[line_id], **instrument)
            else:
                raise ValueError(f"Unknown instrument type: {instrument_type}")

            if velocity_correction:
                velocity_correction_models[instrument_id] = VelocityCorrectionModel()
            if limb_correction:
                limb_correction_models[instrument_id] = LimbCorrectionModel()
            if artifact_correction:
                artifact_correction_models[instrument_id] = ArtifactCorrectionModule()

        self.compile_config = dict(compile_config or {})
        unknown = self.compile_config.keys() - {'me_solver', 'mode'}
        if unknown:
            raise ValueError(f'Unknown compile settings: {sorted(unknown)}')
        enabled = self.compile_config.get('me_solver', False)
        if not isinstance(enabled, bool):
            raise ValueError('compile_config.me_solver must be a boolean.')
        mode = self.compile_config.get('mode', 'default')
        if mode not in ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'):
            raise ValueError(f'Unknown compilation mode: {mode}')
        for forward_model in forward_models.values():
            forward_model._compile_solver_mode = mode if enabled else None

        self.spectral_responses = nn.ModuleDict(spectral_responses)
        self.forward_models = nn.ModuleDict(forward_models)
        self.velocity_correction_models = nn.ModuleDict(velocity_correction_models)
        self.limb_correction_models = nn.ModuleDict(limb_correction_models)
        self.artifact_correction_models = nn.ModuleDict(artifact_correction_models)
        self.hmi_instrument_ids = tuple(hmi_instrument_ids)

        self.lr_params = lr_params
        self.optimizer_config = copy.deepcopy(optimizer_config or {'type': 'soap'})
        self.optimizer_config.setdefault('type', 'soap')

        self.validation_outputs = {}
        normalization_config = normalization_config if normalization_config is not None else {}
        self.normalization = NormalizationModule(**normalization_config)
        stokes_loss_config = copy.deepcopy(stokes_loss_config) if stokes_loss_config is not None else {}
        self.stokes_loss = StokesLossModule(**stokes_loss_config)
        self.stokes_loss_config = self.stokes_loss.configuration()
        # Validate independent data weights and continuous physics constraints.
        weights_config, physics_config = split_stokes_and_physics_config(
            weights_config, physics_config,
        )
        weights_config = {
            component: weights_config.get(component, 1.0)
            for component in ('I', 'Q', 'U', 'V')
        }
        self.stokes_weights_schedules = {
            component: PhysicsWeightSchedule.from_config(value)
            for component, value in weights_config.items()
        }
        self.stokes_weights_config = {
            component: schedule.configuration()
            for component, schedule in self.stokes_weights_schedules.items()
        }
        self.stokes_weights = nn.ParameterDict({
            component: nn.Parameter(torch.tensor(schedule.start), requires_grad=False)
            for component, schedule in self.stokes_weights_schedules.items()
        })
        self.physics_regularization = PhysicsRegularization(
            physics_config,
            physics_domain,
            vector_potential=self.parameter_model.vector_potential,
            gauss_per_dB=gauss_per_dB,
            meters_per_ds=Rs_per_ds * (1 * u.Rsun).to_value(u.m),
            seconds_per_dt=seconds_per_dt,
        )
        self.physics_config = self.physics_regularization.configuration()

        # time shift config
        time_shift_config = {'scale': 0.0, 'iterations': 0.0} if time_shift_config is None else time_shift_config
        scale_value = time_shift_config['scale'] / seconds_per_dt
        self._time_shift_enabled = scale_value > 0
        self.time_shift_scale = nn.Parameter(torch.tensor(scale_value, dtype=torch.float32), requires_grad=False)
        self.time_shift_scale_gamma = scale_value / time_shift_config['iterations'] if time_shift_config['iterations'] > 0 else 0.0
        self.gauss_per_dB = gauss_per_dB
        self.Rs_per_ds = Rs_per_ds
        self.meters_per_ds = Rs_per_ds * (1 * u.Rsun).to_value(u.m)
        self.seconds_per_dt = seconds_per_dt
        self.val_outputs = []

    @torch.no_grad()
    def _set_stokes_weights_step(self, global_step):
        for component, schedule in self.stokes_weights_schedules.items():
            self.stokes_weights[component].fill_(schedule.value_at(global_step))

    def _get_spectral_response(self, batch, instrument_id):
        keys = ('spectral_offsets', 'spectral_weights', 'continuum_weights')
        missing = [key for key in keys if key not in batch]
        if instrument_id in self.hmi_instrument_ids and missing:
            raise KeyError(
                f'HMI batch {instrument_id!r} is missing dataset-sampled response arrays: {missing}.'
            )
        if missing and len(missing) != len(keys):
            raise KeyError(f'Incomplete spectral response for {instrument_id!r}: missing {missing}.')
        if missing:
            return self.default_spectral_response(instrument_id, len(batch['coords']),
                batch['wavelength_grid'].shape[-1], device=batch['coords'].device)
        return {key: batch[key] for key in keys}

    def default_spectral_response(self, instrument_id, n_profiles, n_wavelengths, device=None):
        """Return configured non-HMI passband samples; explicit batch data wins."""
        if instrument_id in self.hmi_instrument_ids:
            raise ValueError('HMI requires acquisition-sampled transmission arrays.')
        if instrument_id not in self.spectral_responses:
            raise KeyError(f'Unknown instrument {instrument_id!r}.')
        return self.spectral_responses[instrument_id](n_profiles, n_wavelengths, device=device)

    def _synthesize_stokes(
        self,
        instrument_id,
        forward_params,
        mu,
        wavelength_grid,
        spectral_response,
        filling_factor=None,
    ):
        """Synthesize one instrument and optionally mix a field-free component."""
        magnetic_components = self.forward_models[instrument_id](
            **forward_params,
            mu=mu,
            wavelength_grid=wavelength_grid,
            **spectral_response,
        )
        stokes = torch.stack(magnetic_components, dim=-2)
        if filling_factor is None:
            return stokes

        field_free_components = self.forward_models[instrument_id](
            **field_free_forward_parameters(forward_params),
            mu=mu,
            wavelength_grid=wavelength_grid,
            **spectral_response,
        )
        field_free_stokes = torch.stack(field_free_components, dim=-2)
        stokes = mix_magnetic_filling_factor(
            stokes,
            field_free_stokes,
            filling_factor,
        )
        return stokes

    def _reduce_stokes_objective(self, wavelength_loss, sample_weights=None):
        """Apply the established wavelength, sample, and Stokes reductions."""
        sample_loss = (wavelength_loss.sum(dim=-1) if self.wavelength_reduction == 'sum'
                       else wavelength_loss.mean(dim=-1))
        if sample_weights is None:
            component_loss = sample_loss.mean(dim=0)
        else:
            sample_weights = sample_weights.reshape(-1).to(sample_loss)
            if sample_weights.shape != (sample_loss.shape[0],):
                raise ValueError('sample_weight must contain one value per Stokes profile.')
            if not torch.isfinite(sample_weights).all() or torch.any(sample_weights <= 0):
                raise ValueError('sample_weight values must be finite and positive.')
            component_loss = (sample_loss * sample_weights[:, None]).sum(0) / sample_weights.sum()
        weights = torch.stack([self.stokes_weights[key] for key in ('I', 'Q', 'U', 'V')])
        total = (torch.einsum('...i,i->...', sample_loss, weights).mean() if sample_weights is None
                 else torch.einsum('i,i->', component_loss, weights))
        return component_loss, total

    def reduce_instrument_losses(self, losses, sample_weights=None):
        """Combine scalar objectives without concatenating spectral arrays.

        Default weighting is by profile count, matching the original objective
        when wavelength counts agree. Explicit instrument weights instead set
        each instrument's relative contribution independently of batch sizes.
        """
        if not losses:
            raise ValueError('At least one instrument loss is required.')
        sample_weights = sample_weights or {}
        reduced = {key: self._reduce_stokes_objective(value, sample_weights.get(key))
                   for key, value in losses.items()}
        weights = {key: self.instrument_weights.get(key, 1.) if self.instrument_weights
                   else (sample_weights[key].sum() if sample_weights.get(key) is not None else value.shape[0])
                   for key, value in losses.items()}
        denominator = sum(weights.values())
        components = sum(reduced[key][0] * weights[key] for key in reduced) / denominator
        objective = sum(reduced[key][1] * weights[key] for key in reduced) / denominator
        return components, objective

    def configure_optimizers(self):
        parameters = (list(self.parameter_model.parameters()) +
                      list(self.forward_models.parameters()) +
                      list(self.velocity_correction_models.parameters()) +
                      list(self.limb_correction_models.parameters()) +
                      list(self.artifact_correction_models.parameters()))
        parameters = [p for p in parameters if p.requires_grad]
        if isinstance(self.lr_params, dict):
            learning_rate = float(self.lr_params['start'])
            end_learning_rate = float(self.lr_params['end'])
            decay_iterations = int(self.lr_params['iterations'])
            if learning_rate <= 0 or end_learning_rate <= 0:
                raise ValueError('Learning rates must be positive.')
            if decay_iterations <= 0:
                raise ValueError('lr_params.iterations must be positive.')
        elif isinstance(self.lr_params, (float, int)):
            learning_rate = float(self.lr_params)
        else:
            raise ValueError(f"Invalid lr_params: {self.lr_params}, must be dict or float/int")

        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('Learning rates must be finite and positive.')
        if isinstance(self.lr_params, dict) and not np.isfinite(end_learning_rate):
            raise ValueError('Learning rates must be finite and positive.')
        optimizer_config = copy.deepcopy(self.optimizer_config)
        optimizer_type = optimizer_config.pop('type')
        optimizer_config.setdefault('weight_decay', 0.0)
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate, **optimizer_config)
        elif optimizer_type == 'soap':
            optimizer = SOAP(parameters, lr=learning_rate, **optimizer_config)
        else:
            raise ValueError(f'Unknown optimizer type: {optimizer_type!r}. Use adamw or soap.')
        if not isinstance(self.lr_params, dict):
            return optimizer

        decay_ratio = end_learning_rate / learning_rate
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: decay_ratio ** min(step / decay_iterations, 1.0),
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_nb):
        self._set_stokes_weights_step(int(self.global_step))
        instrument_ids = list(batch.keys())

        coords = torch.cat([batch[k]['coords'] for k in instrument_ids])
        cartesian_to_spherical_transform = torch.cat(
            [batch[k]['cartesian_to_spherical_transform'] for k in instrument_ids])
        rtp_to_img_transform = torch.cat([batch[k]['rtp_to_img_transform'] for k in instrument_ids])
        v_obs_los = torch.cat([batch[k]['v_obs_los'] for k in instrument_ids])
        ds_lengths = {k: batch[k]['coords'].shape[0] for k in instrument_ids}

        # apply random shift to time coordinate
        if self._time_shift_enabled:
            coords = random_time_shift_coords(coords, self.seconds_per_dt, scale=self.time_shift_scale)

        # forward step
        # Physics collocation points manage their own derivative graph below.
        if self.parameter_model.vector_potential:
            coords.requires_grad_(True)
        output = self.parameter_model(coords)

        #################################################
        # Transform the physical Carrington-frame model outputs to each
        # observer's image frame before Stokes synthesis.
        transformed = self.transform_parameters(
            output, coords, cartesian_to_spherical_transform, rtp_to_img_transform,
        )
        forward_params = self.scale_parameters(output, transformed, v_obs_los)

        current_idx = 0
        instrument_losses = {}
        objective_inputs = {}
        for instrument_id, n_samples in ds_lengths.items():
            ds_mu = batch[instrument_id]['rtp_to_img_transform'][..., 2, 0:1]
            ds_wavelength_grid = batch[instrument_id]['wavelength_grid']
            ds_stokes_true = batch[instrument_id]['stokes']
            ds_spectral_response = self._get_spectral_response(batch[instrument_id], instrument_id)

            ds_forward_params = {
                key: value[current_idx:current_idx + n_samples]
                for key, value in forward_params.items()
            }
            if getattr(self.parameter_model, 'line_heads', None):
                line_output = self.parameter_model(
                    coords[current_idx:current_idx + n_samples], instrument_id=instrument_id)
                for key in ('vmac', 'damping', 'b0', 'b1', 'kl'):
                    ds_forward_params[key] = line_output[key]
            if instrument_id in self.velocity_correction_models:
                time_coords = batch[instrument_id]['coords'][..., 3:4]
                ds_forward_params['vdop'] = ds_forward_params['vdop'] + \
                    self.velocity_correction_models[instrument_id](time_coords)
            limb_correction = None
            if instrument_id in self.limb_correction_models:
                limb_correction = self.limb_correction_models[instrument_id](ds_mu)
                ds_forward_params = self.correct_limb_effects(ds_forward_params, limb_correction)

            ds_stokes_pred = self._synthesize_stokes(
                instrument_id,
                ds_forward_params,
                ds_mu,
                ds_wavelength_grid,
                ds_spectral_response,
                None if limb_correction is None else limb_correction['filling_factor'],
            )
            if instrument_id in self.artifact_correction_models:
                if 'pix' not in batch[instrument_id]:
                    raise KeyError(
                        f"Artifact correction requires 'pix' for instrument {instrument_id}."
                    )
                correction = self.artifact_correction_models[instrument_id](
                    batch[instrument_id]['pix'], ds_stokes_pred
                )
                ds_stokes_pred = correction['stokes_corr']
            instrument_losses[instrument_id] = self.stokes_loss(
                ds_stokes_pred, ds_stokes_true, self.normalization)
            objective_inputs[instrument_id] = {
                'prediction': ds_stokes_pred, 'target': ds_stokes_true,
                'observed_continuum': self.stokes_loss._continuum(ds_stokes_true.detach()),
            }
            current_idx += n_samples

        #################################################
        # Reduce each instrument before combining: wavelength counts may differ.
        component_loss, stokes_total = self.reduce_instrument_losses(
            instrument_losses, {key: batch[key].get('sample_weight') for key in instrument_ids})
        I_loss, Q_loss, U_loss, V_loss = component_loss
        logging_loss = {"I": I_loss, "Q": Q_loss, "U": U_loss, "V": V_loss,
                        "stokes_loss": stokes_total}
        logging_loss.update({
            f"weights.{component}": value.detach()
            for component, value in self.stokes_weights.items()
        })

        physics = self.physics_regularization.evaluate(
            self.parameter_model,
            global_step=int(self.global_step),
            reference=stokes_total,
        )
        total_loss = stokes_total + physics.total
        logging_loss.update(physics.metrics)
        if not torch.isfinite(total_loss):
            detail = write_numerical_report(self, 'forward', {
                'batch': batch, 'decoded': output, 'forward_parameters': forward_params,
                'profiles_by_instrument': objective_inputs,
                'elementwise_loss': instrument_losses, 'stokes_total': stokes_total,
                'physics_total': physics.total, 'total_loss': total_loss,
            })
            raise RuntimeError(f'Encountered non-finite total loss. {detail}')
        logging_loss['loss'] = total_loss
        return logging_loss

    def scale_parameters(self, output, transformed_output, v_obs_los):
        return scale_spherical_forward_parameters(
            output, transformed_output, v_obs_los,
            self.gauss_per_dB, self.meters_per_ds, self.seconds_per_dt,
        )

    def transform_parameters(self, output, coords, cartesian_to_spherical_transform, rtp_to_img_transform):
        return transform_spherical_parameters(
            output, coords, cartesian_to_spherical_transform, rtp_to_img_transform,
            self.meters_per_ds, self.seconds_per_dt,
        )

    def correct_limb_effects(self, parameters, limb_correction):
        return correct_spherical_limb_effects(parameters, limb_correction)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        algorithm = getattr(gradient_clip_algorithm, 'value', gradient_clip_algorithm) or 'norm'
        if gradient_clip_val is None or gradient_clip_val <= 0 or algorithm != 'norm':
            return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        parameters = [parameter for group in optimizer.param_groups for parameter in group['params']
                      if parameter.grad is not None]
        try:
            # Reuse the norm calculation already required for clipping. Stop
            # before NaN/Inf gradients can enter SOAP/Adam moments or weights.
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_val, error_if_nonfinite=True)
        except RuntimeError as error:
            if 'non-finite' not in str(error):
                raise
            detail = write_numerical_report(self, 'backward')
            raise RuntimeError(f'Non-finite gradient norm; optimizer update aborted. {detail}') from error

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # Update augmentation each batch without reading a device scalar.
        if self._time_shift_enabled:
            self.time_shift_scale.sub_(self.time_shift_scale_gamma).clamp_(min=0)
        metrics = {
            key if key.startswith('weights.') else f'train.{key}': value
            for key, value in outputs.items()
        }
        if self._train_metrics.keys is not None and tuple(sorted(metrics)) != self._train_metrics.keys:
            # A physics schedule can activate/deactivate terms mid-window.
            self._flush_train_metrics()
        self._train_metrics.add(metrics)
        if self._train_metrics.count >= self.metric_log_interval:
            self._flush_train_metrics()

    def _flush_train_metrics(self, epoch_end=False):
        metrics = self._train_metrics.drain()
        if metrics:
            metrics['Learning Rate'] = self.optimizers().param_groups[0]['lr']
            metrics['time_shift_scale'] = self.time_shift_scale.detach() * self.seconds_per_dt
            # Already reduced together, including physics metrics and stokes_weights.
            log_options = {'sync_dist': False, 'on_step': not epoch_end,
                           'on_epoch': epoch_end, 'batch_size': 1}
            # Display the same window-averaged objective sent to the logger.
            # Reuse the existing interval rather than logging each batch's loss.
            self.log('train.loss', metrics.pop('train.loss'), prog_bar=True, **log_options)
            self.log_dict(metrics, **log_options)

    def on_train_epoch_end(self):
        # Preserve a short final window when an epoch/run ends before 50 batches.
        self._flush_train_metrics(epoch_end=True)

    @torch.enable_grad()
    def validation_step(self, batch, batch_nb):
        self._set_stokes_weights_step(int(self.global_step))
        coords = batch['coords']
        mu = batch['rtp_to_img_transform'][..., 2, 0:1]
        stokes_true = batch['stokes']
        cartesian_to_spherical_transform = batch['cartesian_to_spherical_transform']
        rtp_to_img_transform = batch['rtp_to_img_transform']
        v_obs_los = batch['v_obs_los']
        wavelength_grid = batch['wavelength_grid']
        instrument_id = batch['instrument_id']
        spectral_response = self._get_spectral_response(batch, instrument_id)

        # forward step
        if self.parameter_model.vector_potential:
            coords.requires_grad_(True)
        output = self.parameter_model(coords, instrument_id=instrument_id)

        transformed_output = self.transform_parameters(output, coords,
                                                       cartesian_to_spherical_transform, rtp_to_img_transform)
        forward_params = self.scale_parameters(output, transformed_output, v_obs_los)

        if instrument_id in self.limb_correction_models:
            limb_correction = self.limb_correction_models[instrument_id](mu)
            forward_params = self.correct_limb_effects(forward_params, limb_correction)
        else:
            limb_correction = None

        v_obs_correction = torch.zeros_like(v_obs_los)
        if instrument_id in self.velocity_correction_models:
            time_coords = coords[..., 3:4]
            v_obs_correction = self.velocity_correction_models[instrument_id](time_coords)
            forward_params['vdop'] += v_obs_correction

        artifact_correction = None
        stokes_pred = self._synthesize_stokes(
            instrument_id,
            forward_params,
            mu,
            wavelength_grid,
            spectral_response,
            None if limb_correction is None else limb_correction['filling_factor'],
        )
        if instrument_id in self.artifact_correction_models:
            if 'pix' not in batch:
                raise KeyError(
                    f"Artifact correction requires 'pix' for instrument {instrument_id}."
                )
            artifact_correction = self.artifact_correction_models[instrument_id](
                batch['pix'], stokes_pred
            )
            stokes_pred = artifact_correction['stokes_corr']

        objective_wavelength = self.stokes_loss(
            stokes_pred, stokes_true, self.normalization,
        )

        # Plot and measure the exact transformed profiles compared by the loss.
        stokes_pred_normalized, stokes_true_normalized = self.stokes_loss.transform_profiles(
            stokes_pred, stokes_true, self.normalization,
        )

        diff = torch.abs(stokes_true_normalized - stokes_pred_normalized)
        objective_components = (objective_wavelength.sum(-1) if self.wavelength_reduction == 'sum'
                                else objective_wavelength.mean(-1))
        objective_weights = torch.stack([
            self.stokes_weights['I'], self.stokes_weights['Q'], self.stokes_weights['U'], self.stokes_weights['V']
        ])
        objective = torch.einsum('...i,i->...', objective_components, objective_weights)
        objective_valid = self.stokes_loss.valid_sample_mask(stokes_true)

        res = {'diff': diff,
               'objective': objective,
               'objective_valid': objective_valid,
               'stokes_true': stokes_true_normalized, 'stokes_pred': stokes_pred_normalized,
               **forward_params,
               'b_rtp': transformed_output['b_rtp'] * self.gauss_per_dB,
               'v_rtp': transformed_output['v_rtp'] * self.meters_per_ds / self.seconds_per_dt,
               'b_img': transformed_output['b_img'] * self.gauss_per_dB,
               'v_img': transformed_output['v_img'] * self.meters_per_ds / self.seconds_per_dt,
               'mu': mu, 'v_obs_los': v_obs_los,
               'v_obs_correction': v_obs_correction,
               }
        if limb_correction is not None:
            res['c_b0'] = limb_correction['c_b0']
            res['c_b1'] = limb_correction['c_b1']
            res['c_vdop'] = limb_correction['c_vdop']
            res['limb_filling_factor'] = limb_correction['filling_factor']
        if artifact_correction is not None:
            res['artifact_correction_params'] = artifact_correction['correction_params']
            res['artifact_gate'] = artifact_correction['correction_gate']

        res["lin_idx"] = batch["lin_idx"].long()  # for assembling full images later

        return res

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        # Only rank-0 needs to keep outputs if you're running val on rank-0 only.
        if outputs is not None:
            # ensure CPU to keep GPU mem low
            cpu_out = {k: v.detach().cpu() for k, v in outputs.items()}
            self.val_outputs.append(cpu_out)

    def on_validation_epoch_end(self):
        outputs_list = self.val_outputs
        self.val_outputs = []
        if not outputs_list or any(len(o) == 0 for o in outputs_list):
            return

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            if rank == 0:
                obj_gather_list = [None] * world
                dist.gather_object(outputs_list, obj_gather_list, dst=0)
                outputs_list = [item for sub in obj_gather_list for item in sub]
            else:  # only log for rank 0
                dist.gather_object(outputs_list, None, dst=0)
                return

        # Each batch covers a contiguous native-image slice. DDP may repeat it.
        batches = {}
        for output in outputs_list:
            index = int(output['lin_idx'].reshape(-1)[0])
            batches.setdefault(index, output)
        outputs_list = [batches[index] for index in sorted(batches)]

        outputs = {}
        for k in outputs_list[0]:
            outputs[k] = torch.cat([o[k] for o in outputs_list], dim=0)

        I_diff, Q_diff, U_diff, V_diff = torch.nanmean(outputs['diff'], dim=(0, 2))
        objective_valid = outputs['objective_valid'].bool()
        if torch.any(objective_valid):
            objective_masked = torch.where(
                objective_valid,
                outputs['objective'],
                torch.zeros_like(outputs['objective']),
            )
            objective_total = objective_masked.sum() / objective_valid.sum()
        else:
            objective_total = torch.tensor(torch.nan, dtype=outputs['objective'].dtype)

        image_logger = self._validation_image_logger()
        full_image = outputs['diff'].shape[0] == int(np.prod(self.image_shape[:2]))
        if image_logger is not None and full_image and torch.any(objective_valid):
            parameters = {}
            for k in ['b_field', 'inc', 'azi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl',
                      'v_rtp', 'b_rtp', 'v_img', 'b_img', 'v_obs_correction',
                      'induction', 'dB_dt', 'curl_VxB', 'divergence', 'force_free', 'dB_dr', 'mu', 'v_obs_los',
                      'c_b0', 'c_b1', 'c_vmac', 'c_damping', 'c_kl', 'c_vdop',
                      'limb_filling_factor']:
                if k not in outputs:
                    continue
                v = outputs[k].reshape(*self.image_shape[:2], -1).numpy()
                parameters[k] = v[..., 0] if v.shape[-1] == 1 else v
            self.plot_parameter_overview(parameters)
            self.plot_B_rtp_log(parameters)
            self.plot_B_rtp(parameters)
            self.plot_v_rtp(parameters)
            # self.plot_physics_losses(parameters)
            self.plot_limb_correction(parameters)

            if 'artifact_correction_params' in outputs:
                artifact_params = outputs['artifact_correction_params'].reshape(*self.image_shape[:2], -1).cpu().numpy()
                self.plot_artifact_corrections(artifact_params)

            stokes_true = outputs['stokes_true'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)
            stokes_pred = outputs['stokes_pred'].cpu().numpy().reshape(*self.image_shape[:2], 4, -1)

            self.plot_stokes(stokes_pred, stokes_true)
            self.plot_profile(stokes_pred, stokes_true)

        validation_metrics = {
            # log total stokes loss
            "valid.diff": torch.nanmean(outputs['diff']),
            # log stokes differences
            'valid.I': I_diff, 'valid.Q': Q_diff, 'valid.U': U_diff, 'valid.V': V_diff,
            # exact selected objective, using the same wavelength and Stokes
            # reductions as the training Stokes term
            'valid.objective': objective_total,
            # log physics losses
            # 'valid.induction': torch.nanmean(outputs['induction']),
            # 'valid.divergence': torch.nanmean(outputs['divergence']),
            # 'valid.force_free': torch.nanmean(outputs['force_free']),
        }
        self.log_dict(validation_metrics)

    def _validation_image_logger(self):
        if not self.plot_validation or self._trainer is None or self.trainer.sanity_checking:
            return None
        return next((logger for logger in self.trainer.loggers if isinstance(logger, WandbLogger)), None)

    def _log_validation_image(self, figure, name):
        logger = self._validation_image_logger()
        if logger is not None:
            log_wandb_image(figure, name, logger=logger, step=int(self.global_step))

    def plot_physics_losses(self, parameters):
        induction = parameters['induction']
        dB_dt = parameters['dB_dt']
        curl_VxB = parameters['curl_VxB']
        divergence = parameters['divergence']
        dB_dr = parameters['dB_dr']

        fig, axs = plt.subplots(3, 3, figsize=(10, 6), dpi=100)
        ax = axs[0, 0]
        im = ax.imshow(induction, origin='lower', norm='log')
        ax.set_title("Induction")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(dB_dt, origin='lower', norm='log')
        ax.set_title("dB/dt")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(curl_VxB, origin='lower', norm='log')
        ax.set_title("Curl VxB")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(divergence, origin='lower', norm='log')
        ax.set_title("Divergence")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(dB_dr, origin='lower', norm='log')
        ax.set_title("dB/dr")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        ax.set_axis_off()

        ax = axs[2, 0]
        im = ax.imshow(parameters['force_free'], origin='lower', norm='log')
        ax.set_title("force_free")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 1]
        ax.set_axis_off()

        ax = axs[2, 2]
        ax.set_axis_off()

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self._log_validation_image(fig, 'Physics Losses')
        plt.close(fig)

    def plot_limb_correction(self, outputs):
        required = {'c_b0', 'c_b1', 'c_vdop', 'limb_filling_factor'}
        if not required <= outputs.keys():
            return
        c_b0 = outputs['c_b0']
        c_b1 = outputs['c_b1']
        mu = outputs['mu']
        c_vdop = outputs['c_vdop']
        filling_factor = outputs['limb_filling_factor']

        fig, axs = plt.subplots(1, 5, figsize=(12.5, 2.5), dpi=100)

        ax = axs[0]
        im = ax.imshow(mu, origin='lower', cmap='cividis', vmin=0, vmax=1)
        ax.set_title("mu")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[4]
        im = ax.imshow(filling_factor, origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_title(r"$f(\mu)$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1]
        im = ax.imshow(c_b0, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{b0}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2]
        im = ax.imshow(c_b1, origin='lower', cmap='magma')
        ax.set_title(r"$c_\text{b1}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[3]
        v_min_max = np.nanmax(np.abs(c_vdop))
        im = ax.imshow(c_vdop, origin='lower', cmap='RdBu_r', vmin=-v_min_max, vmax=v_min_max)
        ax.set_title(r"$c_\text{vdop}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        fig.suptitle("Limb Correction Factors")
        fig.tight_layout()
        self._log_validation_image(fig, "Limb Correction Factors")
        plt.close(fig)

    def plot_profile(self, stokes_pred, stokes_true):
        """Spatial mean/std and GT polarization extrema in loss-transformed units."""
        valid = (np.isfinite(stokes_true).all(axis=(-2, -1))
                 & np.isfinite(stokes_pred).all(axis=(-2, -1)))
        positions = np.argwhere(valid)
        if not len(positions):
            return
        true_profiles = stokes_true[valid]
        pred_profiles = stokes_pred[valid]
        # One joint Q/U/V score selects the same native pixel for all four rows.
        polarization = np.sqrt(np.mean(np.square(true_profiles[:, 1:].astype(np.float64)), axis=(1, 2)))
        extrema = [positions[np.argmax(polarization)], positions[np.argmin(polarization)]]
        samples = np.arange(stokes_true.shape[-1])
        fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex=True, squeeze=False)
        for row, label in enumerate('IQUV'):
            ax = axes[row, 0]
            for profiles, color, shade, name in (
                    (true_profiles, 'black', 'gray', 'GT'),
                    (pred_profiles, 'tab:orange', 'tab:orange', 'SPINNME')):
                mean = profiles[:, row].mean(axis=0)
                std = profiles[:, row].std(axis=0)
                ax.fill_between(samples, mean - std, mean + std, color=shade, alpha=0.2)
                ax.plot(samples, mean, color=color, label=name, linewidth=1.6)
            for column, (y, x) in enumerate(extrema, start=1):
                ax = axes[row, column]
                ax.plot(samples, stokes_true[y, x, row], '.-', color='black', label='GT')
                ax.plot(samples, stokes_pred[y, x, row], '-', color='tab:orange', label='SPINNME')
            for ax in axes[row]:
                ax.set_ylabel(label)
                ax.grid(alpha=0.2)
                ax.spines[['top', 'right']].set_visible(False)
                if row == 3:
                    ax.set_xlabel('Wavelength sample')
        axes[0, 0].set_title(f'Mean ± 1 std ({len(positions)} pixels)')
        axes[0, 0].legend(frameon=False)
        for column, ((y, x), title) in enumerate(zip(extrema, ('Strongest polarization', 'Minimum polarization')), start=1):
            axes[0, column].set_title(f'{title}\nx={x}, y={y}')
        fig.suptitle('Stokes profiles - Comparison')
        fig.tight_layout(rect=(0, 0.025, 1, 0.96))
        fig.text(0.5, 0.01, 'Polarization: RMS of GT Q, U, V across wavelength · profiles in loss-transformed units',
                 ha='center', fontsize=9, color='0.35')
        self._log_validation_image(fig, 'Stokes profiles - Comparison')
        plt.close(fig)

    def plot_stokes(self, stokes_pred, stokes_true):
        # plot comparison of integrated stokes vectors
        integerated_stokes_true = np.abs(stokes_true).sum(-1)
        integerated_stokes_pred = np.abs(stokes_pred).sum(-1)
        fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
        for i, label in enumerate(['I', 'Q', 'U', 'V']):
            v_min = np.nanmin(integerated_stokes_true[:, :, i])
            v_max = np.nanmax(integerated_stokes_true[:, :, i])
            norm = ImageNormalize(vmin=v_min, vmax=v_max)
            im = ax[0, i].imshow(integerated_stokes_true[:, :, i], norm=norm, origin='lower')
            ax[0, i].set_title(f"true - {label}")
            divider = make_axes_locatable(ax[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
            im = ax[1, i].imshow(integerated_stokes_pred[:, :, i], norm=norm, origin='lower')
            ax[1, i].set_title(f"pred - {label}")
            divider = make_axes_locatable(ax[1, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        # fig.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self._log_validation_image(fig, "Integrated Stokes vector - Comparison")
        plt.close(fig)

    def plot_parameter_overview(self, parameters):
        b = parameters['b_field']
        inc = parameters['inc']
        azi = parameters['azi']
        # reproject vectors (inc flip with negative B)
        b_xyz = to_cartesian(b, inc, azi)
        b, inc, azi = to_spherical(b_xyz)
        inc = inc % np.pi
        azi = azi % (2 * np.pi)

        fig, axs = plt.subplots(2, 5, figsize=(16, 4), dpi=100)
        ax = axs[0, 0]
        im = ax.imshow(b, cmap='viridis', vmin=.1, origin='lower', norm='log')
        ax.set_title("B")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 1]
        im = ax.imshow(inc % np.pi, cmap='seismic', vmin=0, vmax=np.pi, origin='lower')
        ax.set_title("Inclination")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 2]
        im = ax.imshow(azi % (2 * np.pi), cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower')
        ax.set_title('Azimuth')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 3]
        im = ax.imshow(parameters['b0'], origin='lower')
        ax.set_title("B0")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[0, 4]
        im = ax.imshow(parameters['b1'], origin='lower')
        ax.set_title("B1")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 0]
        im = ax.imshow(parameters['vmac'], origin='lower')
        ax.set_title("Vmac")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 1]
        im = ax.imshow(parameters['damping'], origin='lower')
        ax.set_title("Damping")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 2]
        ax.set_axis_off()
        ax = axs[1, 3]
        vdop_max = np.nanmax(np.abs(parameters['vdop']))
        im = ax.imshow(parameters['vdop'], cmap='seismic_r', vmin=-vdop_max, vmax=vdop_max, origin='lower')
        ax.set_title("Vdop")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        ax = axs[1, 4]
        im = ax.imshow(parameters['kl'], origin='lower')
        ax.set_title("Kl")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self._log_validation_image(fig, "Parameter Overview")
        plt.close(fig)

    def plot_B_rtp_log(self, parameters):
        b_rtp = parameters['b_rtp']
        b_img = parameters['b_img']

        norm = SymLogNorm(linthresh=100, vmin=-5000, vmax=5000)

        fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=100)

        ax = axs[0, 0]
        im = ax.imshow(b_rtp[..., 0], norm=norm, origin='lower', cmap='seismic')
        ax.set_title("$B_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(b_rtp[..., 1], cmap='seismic', norm=norm, origin='lower')
        ax.set_title("$B_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(b_rtp[..., 2], cmap='seismic', norm=norm, origin='lower')
        ax.set_title("$B_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(b_img[..., 0], norm=norm, origin='lower', cmap='seismic')
        ax.set_title(r"$B_\text{xi}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(b_img[..., 1], norm=norm, origin='lower', cmap='seismic')
        ax.set_title(r"$B_\text{eta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        im = ax.imshow(b_img[..., 2], norm=norm, origin='lower', cmap='seismic')
        ax.set_title(r"$B_\text{zeta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self._log_validation_image(fig, 'B')
        plt.close(fig)

    def plot_B_rtp(self, parameters):
        b_rtp = parameters['b_rtp']

        norm = Normalize(vmin=-500, vmax=500)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3), dpi=100)

        ax = axs[0]
        im = ax.imshow(b_rtp[..., 0], norm=norm, origin='lower', cmap='gray')
        ax.set_title("$B_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1]
        im = ax.imshow(b_rtp[..., 1], cmap='gray', norm=norm, origin='lower')
        ax.set_title("$B_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2]
        im = ax.imshow(b_rtp[..., 2], cmap='gray', norm=norm, origin='lower')
        ax.set_title("$B_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self._log_validation_image(fig, 'B_rtp')
        plt.close(fig)

    def plot_v_rtp(self, parameters):
        v_rtp = parameters['v_rtp']
        v_img = parameters['v_img']
        v_obs_correction = parameters['v_obs_correction']

        v_min_max = 5000  # m/s
        norm = Normalize(vmin=-v_min_max, vmax=v_min_max)

        fig, axs = plt.subplots(3, 3, figsize=(10, 7), dpi=100)
        ax = axs[0, 0]
        im = ax.imshow(v_rtp[..., 0], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title("$v_r$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 1]
        im = ax.imshow(v_rtp[..., 1], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title("$v_t$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[0, 2]
        im = ax.imshow(v_rtp[..., 2], cmap='seismic_r', origin='lower', norm=norm)
        ax.set_title("$v_p$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 0]
        im = ax.imshow(v_img[..., 0], cmap='RdBu', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{xi}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 1]
        im = ax.imshow(v_img[..., 1], cmap='RdBu', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{eta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[1, 2]
        im = ax.imshow(v_img[..., 2], cmap='RdBu', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{zeta}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 0]
        im = ax.imshow(parameters['v_obs_los'], cmap='RdBu_r', origin='lower', norm=norm)
        ax.set_title(r"$v_\text{OBS LOS}$")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 1]
        im = ax.imshow(v_obs_correction, cmap='RdBu_r', origin='lower', norm=norm)
        ax.set_title(r'$v_\text{OBS correction}$')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = axs[2, 2]
        ax.set_axis_off()

        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.25, hspace=0.25)
        self._log_validation_image(fig, "v")
        plt.close(fig)

    def plot_artifact_corrections(self, artifact_params):
        """Show active I-to-Q/U/V leakage and its correction gate."""
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for index, label in enumerate(('I → Q', 'I → U', 'I → V', 'Correction gate')):
            values = artifact_params[..., index]
            if index < 3:
                limit = max(float(np.nanmax(np.abs(values))), 1e-8)
                artist = axes[index].imshow(values, origin='lower', cmap='BrBG', vmin=-limit, vmax=limit)
            else:
                artist = axes[index].imshow(values, origin='lower', cmap='viridis', vmin=0, vmax=1)
            axes[index].set_title(label)
            fig.colorbar(artist, ax=axes[index])
        fig.tight_layout()
        self._log_validation_image(fig, 'Artifact leakage and correction gate')
        plt.close(fig)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint['spinnme_checkpoint_version'] = 1
        checkpoint['coordinate_order'] = 'xyzt'
        checkpoint['spinnme_wavelength_reduction'] = self.wavelength_reduction
        checkpoint['spinnme_instrument_weights'] = copy.deepcopy(self.instrument_weights)
        checkpoint['spinnme_optimizer_type'] = self.optimizer_config['type']
        checkpoint['spinnme_stokes_loss_config'] = copy.deepcopy(self.stokes_loss_config)
        if self.stokes_loss.continuum_normalization:
            checkpoint['spinnme_continuum_normalization_mode'] = 'per_profile'
        checkpoint['spinnme_stokes_weights_config'] = copy.deepcopy(
            self.stokes_weights_config
        )
        checkpoint['spinnme_normalization_config'] = self._normalization_objective_config(
            self.normalization.asinh_alphas, self.normalization.stokes_scales
        )
        checkpoint['spinnme_physics_config'] = copy.deepcopy(self.physics_config)

    @staticmethod
    def _normalization_objective_config(asinh_alphas, stokes_scales=None):
        config = {}
        if stokes_scales is not None:
            scales = NormalizationModule(stokes_scales=stokes_scales).stokes_scales
            config['stokes_scales'] = scales.detach().cpu().reshape(-1).tolist()
        if asinh_alphas is None:
            return {**config, 'asinh_alphas': None}
        if isinstance(asinh_alphas, dict):
            values = [asinh_alphas[key] for key in ('Q', 'U', 'V')]
        elif isinstance(asinh_alphas, (float, int)):
            values = [asinh_alphas] * 3
        else:
            values = torch.as_tensor(asinh_alphas).detach().cpu().reshape(-1).tolist()
        if len(values) != 3:
            raise ValueError('Normalization objective must provide Q, U, and V asinh alphas.')
        return {**config, 'asinh_alphas': [float(value) for value in values]}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        required = {
            'spinnme_checkpoint_version', 'coordinate_order', 'state_dict',
            'spinnme_wavelength_reduction', 'spinnme_instrument_weights',
            'spinnme_optimizer_type', 'spinnme_stokes_loss_config',
            'spinnme_stokes_weights_config', 'spinnme_normalization_config',
            'spinnme_physics_config',
        }
        missing = required.difference(checkpoint)
        if missing or checkpoint.get('spinnme_checkpoint_version') != 1:
            raise RuntimeError(
                'Unsupported SPINNME training checkpoint metadata. '
                f'Missing fields: {sorted(missing)}. Use a fresh path.'
            )
        if checkpoint['coordinate_order'] != 'xyzt':
            raise ValueError('Checkpoint coordinate order is incompatible with [x, y, z, t]. Use a fresh path.')
        if checkpoint['spinnme_wavelength_reduction'] != self.wavelength_reduction:
            raise ValueError('Cannot resume with a different wavelength reduction. Use a fresh path.')
        if checkpoint['spinnme_instrument_weights'] != self.instrument_weights:
            raise ValueError('Cannot resume with different instrument weights. Use a fresh path.')
        self._train_metrics.clear()
        state_dict = checkpoint['state_dict']
        self._time_shift_enabled = bool(state_dict['time_shift_scale'].detach().cpu() > 0)
        if (checkpoint.get('optimizer_states')
                and checkpoint['spinnme_optimizer_type'] != self.optimizer_config['type']):
            raise ValueError('Cannot resume with a different optimizer. Use a fresh path.')
        checkpoint_loss_config = dict(checkpoint['spinnme_stokes_loss_config'])
        if (checkpoint_loss_config.get('continuum_normalization')
                and (checkpoint.get('spinnme_continuum_normalization_mode') != 'per_profile'
                     or 'continuum_quantile' not in checkpoint_loss_config)):
            raise RuntimeError(
                'Checkpoint per-profile continuum objective metadata are incomplete. '
                'Use a fresh path.'
            )
        try:
            checkpoint_loss_config = StokesLossModule(**checkpoint_loss_config).configuration()
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                'This checkpoint uses an unsupported Stokes objective. Use a fresh path.'
            ) from error
        saved_normalization = checkpoint['spinnme_normalization_config']
        checkpoint_normalization_config = self._normalization_objective_config(
            saved_normalization.get('asinh_alphas'), saved_normalization.get('stokes_scales'),
        )
        current_normalization_config = self._normalization_objective_config(
            self.normalization.asinh_alphas, self.normalization.stokes_scales
        )

        if (checkpoint_loss_config != self.stokes_loss_config
                or checkpoint_normalization_config != current_normalization_config):
            raise RuntimeError(
                'The configured Stokes objective does not match the checkpoint objective. '
                f'Checkpoint loss/normalization: {checkpoint_loss_config}/'
                f'{checkpoint_normalization_config}; current: {self.stokes_loss_config}/'
                f'{current_normalization_config}. '
                'Use a fresh path when changing the objective transform.'
            )

        checkpoint_weights_config = checkpoint['spinnme_stokes_weights_config']
        if checkpoint_weights_config != self.stokes_weights_config:
            warnings.warn(
                'The configured Stokes-weight schedules differ from the checkpoint; '
                'the configured schedules will replace the checkpoint weights and '
                'continue from its global step. '
                f'Checkpoint: {checkpoint_weights_config}; '
                f'configured: {self.stokes_weights_config}.',
                UserWarning,
                stacklevel=2,
            )
        checkpoint_step = int(checkpoint.get('global_step', 0))
        for component, schedule in self.stokes_weights_schedules.items():
            key = f'stokes_weights.{component}'
            configured_value = schedule.value_at(checkpoint_step)
            reference = state_dict.get(key, self.stokes_weights[component].detach())
            state_dict[key] = reference.new_tensor(configured_value)
        checkpoint['spinnme_stokes_weights_config'] = copy.deepcopy(
            self.stokes_weights_config
        )

        checkpoint_physics_config = checkpoint['spinnme_physics_config']
        if checkpoint_physics_config != self.physics_config:
            warnings.warn(
                'The configured physics objective or collocation domain differs from '
                'the checkpoint; the current configuration will replace the checkpoint '
                'settings and continue from its global step. '
                f'Checkpoint: {checkpoint_physics_config}; '
                f'configured: {self.physics_config}.',
                UserWarning,
                stacklevel=2,
            )
        checkpoint['spinnme_physics_config'] = copy.deepcopy(self.physics_config)

        self.load_state_dict(state_dict, strict=True)
        self.validation_outputs = {}  # reset validation outputs
