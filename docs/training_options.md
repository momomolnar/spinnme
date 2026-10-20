# Customize training

Copy a preset with `spinnme config copy hmi/full_disk --output inversion.yaml`. Edit the matching settings below while retaining your input and instrument configuration. Use a fresh output directory when changing the model, data or objective.

For your first custom run: verify reader axes, wavelengths, normalization and response; choose a small region/time selection; check a short fit; then expand the data/budget and adjust model capacity if needed. Use the [Stokes-fit procedure](evaluation.md#check-the-stokes-fit-before-comparing-maps) for that decision.

## Set the budget and optimizer

```yaml
training:
  epochs: -1
  max_steps: 20000
  lr_params: {start: 0.001, end: 0.0001, iterations: 20000}
  optimizer_config: {type: soap}
```

`max_steps` counts optimizer updates; `epochs` limits passes through the sampled inventory. Each is `-1` or a positive integer, with at least one positive. When both are positive, the first limit reached stops training. Keep step budgets fixed when comparing datasets of different sizes.

The learning rate decays exponentially over `iterations`, then stays at `end`. Use `lr_params: 0.001` for a constant rate. Set the rate here, rather than adding `lr` to `optimizer_config`.

| Optimizer | Choice |
| --- | --- |
| `soap` | Used by the mission presets |
| `adamw` | Simple alternative for a trial |

Both default to zero weight decay. To change it, use, for example, `optimizer_config: {type: adamw, weight_decay: 0.01}`.

## Adjust memory and data loading

```yaml
data:
  batch_size: 4096
  dataset_batch_size: 512
  num_workers: 4
  num_dataset_workers: 8
training:
  accumulate_grad_batches: 1
```

`batch_size` requests profiles per training batch on each device; `dataset_batch_size` is the chunk drawn from one acquisition. Require a positive multiple of the chunk size, or of `dataset_batch_size × number of training entries` when balancing entries. Short final batches are retained.

Reduce batch size for less device memory use. `accumulate_grad_batches: 4` combines four batches per optimizer update; keep this fixed in step-budget comparisons. More `num_workers` can improve training throughput. `num_dataset_workers` defaults to 8 and prepares up to 8 acquisitions concurrently: reading Stokes data, computing geometry, normalizing/filtering and writing caches. `num_workers` separately controls training/validation DataLoader workers. Preparation concurrency is capped by the number of acquisitions; reduce it when RAM is limited. Workers use `spawn` to avoid inheriting background-thread locks from W&B, Torch or Zarr. Set both counts to `0` when debugging. When calling `run()` from a Python script with workers enabled, put the call inside `if __name__ == "__main__":`; use zero workers for an interactive Python session.

By default, Lightning detects an available GPU accelerator and uses its available devices; CPU is the fallback. YAML `training.devices` can restrict that selection. `--device cpu` optionally forces one CPU device. For a specific multi-GPU selection, set `training.accelerator: gpu` and `training.devices: [0, 1]`.

## Change validation frequency

```yaml
training:
  check_val_every_n_epoch: null
  val_check_interval: 500
  metric_log_interval: 50
  log_every_n_steps: 50
  plot_validation: true
```

This validates every 500 training batches across epochs and averages training metrics over 50 batches. These intervals count training batches, including gradient accumulation batches. At every validation, W&B receives magnetic/velocity maps, ME parameter maps, observed versus synthesized Stokes maps and representative I/Q/U/V profiles. Limb/artifact correction maps appear when those modules are enabled. Images use `trainer/global_step` as their training-step axis. `plot_validation` defaults to `true`; set it to `false` to retain only scalar metrics. Rendering is skipped when W&B logging is disabled.

The progress bar displays the averaged total objective as `train.loss` whenever the metric window is flushed, including a short final window at epoch end. `log_every_n_steps` separately controls Lightning's logger cadence in optimizer steps. Keep it aligned with `metric_log_interval` when `accumulate_grad_batches: 1`; for example, use `10` for both to refresh the progress-bar loss and W&B curves more frequently. `training.enable_progress_bar: false` hides the bar without disabling metric logging.

For validation at each epoch's end, use `check_val_every_n_epoch: 1` and `val_check_interval: 1.0`. Keep the decimal point: integer `1` means every batch.

## Change model capacity

Increase MLP width (`dim`) or depth (`n_layers`) when a smaller model underfits, checking the cost in time and memory:

```yaml
model:
  type: mlp
  dim: 128
  n_layers: 4
  activation: swish
  encoding_config:
    type: fourier
    num_frequencies: [8, 8, 8, 4]
    max_frequencies: [64, 64, 64, 4]
```

Fourier lists follow `[x, y, z, time]` in normalized coordinates. Assess higher frequencies against spatial resolution, cadence and noise. Other MLP encodings are `identity`, `gaussian` and `positional_isotropic`.

Coordinates are Carrington-frame x/y/z on the solar sphere, in solar radii divided by `data.Rs_per_ds`; time is seconds from `data.ref_time` divided by `data.seconds_per_dt`. With `Rs_per_ds: 1` and `seconds_per_dt: 3600`, one spatial unit is a solar radius and one time unit is an hour. These internal coordinates support spherical inversion; they do not select a Cartesian inversion mode. Changing these scales also changes the physical meaning of encoding settings.

| Representation | Scale controls |
| --- | --- |
| `identity` | Uses normalized coordinates directly |
| `fourier` | `min_frequency` / `max_frequencies` are direct frequency factors; phases include π |
| `gaussian` | `scale` sets the fixed random projection bandwidth per normalized coordinate; phases include 2π |
| `positional_isotropic` | Spatial directions with separate space/time blocks; `min_frequencies` / `max_frequencies` are base-two exponents, with π in the phases |
| SIREN | `input_omegas` scales normalized coordinates before the learned first layer; `hidden_omega_0` scales hidden sine layers |

These settings change the variation the network can readily represent; they do not define measured spatial or temporal resolving power.

To use a SIREN, replace the whole model section, removing MLP-only activation and encoding settings:

```yaml
model:
  type: siren
  dim: 128
  n_layers: 4
  input_omegas: [30, 30, 30, 3]
  hidden_omega_0: 1.0
  output_init_scale: 0.1
```

## Normalize profiles and weight I/Q/U/V

```yaml
stokes_loss:
  type: mse
  continuum_normalization: true
  continuum_quantile: 0.9
normalization:
  asinh_alphas: {Q: 0.01, U: 0.01, V: 0.01}
weights: {I: 1.0, Q: 1.0, U: 1.0, V: 1.0}
```

The observed and synthesized profiles are each divided by their own intensity quantile. Asinh scaling then increases sensitivity to weak polarization. Inspect residuals and noise when changing `asinh_alphas`.

For linear scaling, replace the entire normalization section:

```yaml
normalization:
  stokes_scales: [1.0, 0.01, 0.01, 0.01]
```

These scales divide I/Q/U/V before the squared residual. Do not combine them with `asinh_alphas`. The objective is transformed MSE, with no automatic per-sample noise-variance likelihood or uncertainty intervals. Equal component weights are not equal noise weights. `weights` adds nonnegative loss weights, separate from wavelength sampling; `{I: 1, Q: 2, U: 2, V: 1}` doubles the Q/U contributions.

For absolute-intensity fits, set `continuum_normalization: false`. If your YAML explicitly sets `model.source_function`, remove it or set it to `independent`. Dataset `stokes_normalization` separately divides the input observations; use the same value for all acquisitions and validation under one instrument ID.

## Choose a validation observation or holdout

`data.valid_config.sample_idx` selects one acquisition, starting at zero; omitting it selects the middle observation. For a holdout from a shared input inventory with at least six acquisitions and no other subsampling:

Set `exclude_samples: [5]` in each matching training dataset and `sample_idx: 5` in `data.valid_config` in your run YAML.

Exclude the observation from every training entry containing it, or use separate training/validation paths. Validation must match a trained instrument and its Stokes normalization. Exclusion indices refer to the time-selected inventory.

## Combine datasets or instruments

Use [custom inversions](custom.md) for reader/line setup and the `generic/multiple_lines` preset as a starting point. Shared instrument IDs require compatible line centers, wavelength counts and intensity scales; distinct IDs may use different layouts.

For equal contributions from that preset's two instruments, independent of sampled profile counts:

```yaml
training:
  wavelength_reduction: mean
  instrument_weights: {LINE6173: 1.0, LINE6302: 1.0}
```

`mean` averages over wavelengths; the default `sum` adds them. Without instrument weights, effective profile counts determine the mixture. Each dataset's `loss_weight` supplies relative per-profile weighting.

## Oversample polarized profiles

Uniform sampling is the default. In the HMI preset, replace the training entries with the two pools below, retaining its validation and instrument sections:

```yaml
data:
  balance_train_configs: true
  batch_size: 8192
  dataset_batch_size: 512
  train_configs:
    - &hmi
      type: hmi
      data_path: ./data/hmi/stokes/*.fits
      transmission_profile_directory: ./data/hmi/transmission
      instrument_id: HMI
      stokes_normalization: 50000.0
    - <<: *hmi
      polarization_filter: {smoothing_sigma: 5, fraction: 0.1, radius_limit: 0.95}
```

Half the batch comes from each entry. The second selects stronger polarization using a smoothed score; selected Stokes profiles stay unchanged. HMI selection requires native-resolution images. Compare oversampling as a separate experiment.

## Profile or compile a run

Use `spinnme run --config inversion.yaml --output runs/profile --profile` to write a bounded timing report under `profiling/`. Allow at least 32 training batches for its skipped, warm-up and active periods.

To try compilation of the profile solver, add:

```yaml
training:
  compile_config: {me_solver: true, mode: default}
```

Compilation adds startup cost. Compare with the default eager solver using the same fit settings, especially for small or changing batch shapes. Snapshot inference remains eager.

## Change spectral sampling

Spectral sampling is separate from image `resolution` and Stokes loss `weights`. See [HMI transmission integration](hmi.md#increase-or-decrease-numerical-sampling) for numerical passband controls, and [Hinode measured wavelengths](hinode.md#measured-wavelength-sampling) and [instrumental response](hinode.md#instrumental-spectral-response) for channel selection and convolution settings.

## Physics constraints and normalization

`physics` adds an independent collocation objective to the Stokes inversion.
Coordinates are Cartesian Carrington coordinates `x_model = x / L_model`,
with `L_model = Rs_per_ds * R_sun`; `B_model = B / gauss_per_dB` and
`t_model = t / seconds_per_dt`. Velocities are in `L_model / seconds_per_dt`.
Derivatives must be taken with respect to those same coordinates.

For `normalization: characteristic`, specify fixed physical units in
`physics.characteristic_units`: `length_Mm`, `magnetic_field_G`,
`velocity_km_s`, and `time_s`. The implementation rescales fields and applies
`d/dx_hat = (L0/L_model) d/dx_model` and the equivalent time factor before
forming residuals. Thus the divergence loss is `(L0/B0)^2 * (div B)^2`, and
the force-free loss is
`(L0/B0)^2 * |curl(B) x B|^2 / (|B|^2 + (epsilon*B0)^2)`
when B and derivatives in these expressions use physical units.
The curl here omits the permeability factor, which is absorbed in the loss
weight. The force-free term penalizes perpendicular current; parallel current
is allowed. It is not normalized by current magnitude.

`raw` losses depend on internal dataset units. `relative` normalizes divergence
and potential residuals by the squared spatial magnetic Jacobian; it retains
the same field-normalized force-free loss as `raw`. Consequently `relative`
is not a substitute for fixed physical characteristic units. Physics weights
set the balance against the continuum/asinh-transformed Stokes MSE and should
be assessed from fit residuals and physics metrics.

For a divergence-free field throughout a shell, use
`model.vector_potential: true` and `model.b_sinh_scale: null`. The model then
returns `B = curl(A)`, so divergence vanishes analytically up to derivative
roundoff. No divergence penalty is needed. A force-free penalty is still a
soft constraint, and its finite sampled loss must be inspected after training.
Use smooth activations such as `swish` because force-free training through A
requires second spatial derivatives and their parameter gradients.

Set `physics.radius_range_Rs: [1.0, 1.1]` explicitly: omitting it uses the
observed radial bounds, usually only the surface. `domain: full_train` covers
the combined angular footprint and time range of the observations, not the
whole solar sphere. `radial_measure: volume` samples uniformly in volume
(uniform longitude, sin(latitude), and radius cubed). Data continue to constrain
the photosphere while physics constrains the shell. These constraints alone
do not specify a unique coronal solution or impose upper/side boundary data.
