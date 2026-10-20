# Train an inversion

Each run fits a new coordinate network to the selected scene or time sequence; [the model overview](method.md) explains its inputs, outputs and assumptions.

Choose the example for your observations—[HMI](hmi.md), [Hinode](hinode.md), [PHI](phi.md), or [custom FITS](custom.md)—and save its configuration as `inversion.yaml`. Use [training customization](training_options.md) when you need to change the model, sampling, loss or execution settings.

## Run your YAML

For example, copy the HMI preset after its observations and transmission profiles are available:

```bash
spinnme config copy hmi/full_disk --output inversion.yaml
# Edit the model/training settings in inversion.yaml.
spinnme config validate inversion.yaml
spinnme run --config inversion.yaml --data_path ./data/hmi \
  --run_path ./runs/my-inversion
```

Copying preserves `<<KEY>>` placeholders. Supply every remaining key as `--KEY VALUE`, or replace it with a concrete YAML value. `--run_path` defaults `work_path` to its `cache/` subdirectory. Validation checks structure without filling inputs or opening files. A YAML with concrete paths can instead run with just `--config inversion.yaml`; see [named values](configuration.md#overwrite-named-values).

`run` builds or reuses data caches automatically. It detects an available GPU accelerator and falls back to CPU. `--device cpu` optionally forces CPU execution. HMI transmission profiles must already be downloaded; Hinode slit assembly is a separate preprocessing operation. The instrument guides show these steps.

To inspect and cache observations before a long run, optionally use `spinnme prep --config inversion.yaml --data_path ./data/hmi --run_path ./runs/my-inversion`. This writes `preparation.json`; it does not download files, modify the YAML or set up a new training config. Run the same `inversion.yaml` afterward.

Each training batch samples observed profiles, evaluates the shared magnetic and velocity fields, synthesizes the instrument's Stokes profiles, and updates the model from the residuals. Training stops at the configured step or epoch limit.

## Monitor the fit

The console reports preparation, the selected device and training progress. The progress bar shows `train.loss`, the total training objective averaged over `training.metric_log_interval` batches (50 by default), including any enabled physics penalties. It first appears after that interval, or at the end of a shorter epoch, and works with logging disabled. Enable [Weights & Biases](#weights--biases) to inspect training and validation curves. Important metrics include `train.loss`, `train.stokes_loss`, the I/Q/U/V component losses, and `valid.objective`.

Validation evaluates the single observation selected by `data.valid_config`. When that observation also appears in training, its loss measures the fit to training data. For an independent check, [exclude it from training](training_options.md#choose-a-validation-observation-or-holdout).

For a practical procedure, follow [check the Stokes fit](evaluation.md#check-the-stokes-fit-before-comparing-maps). Validation metrics and images use the configured continuum normalization and asinh/linear transformation; they are not automatically reduced chi-squared or physical field errors. The bundled Hinode validation covers 6302, so assess 6301 separately too.

Set `training.val_check_interval` in your YAML to control validation frequency. For a short trial, edit the training budget and validation interval before running into a fresh output directory; see [training options](training_options.md).

## Save, export and resume

A successful run writes these files under its output directory:

| File | Use |
| --- | --- |
| `inversion.pme` | Load the fitted model and export maps |
| `last.ckpt` | Resume training with optimizer state |
| `resolved.yaml` | Inspect the settings used for the run |
| `run_manifest.json` | Check completion status and the completed step count |

The inference snapshot is updated after validation and at successful training completion, including runs that finish before their first validation. Export it with:

```bash
spinnme export runs/my-inversion/inversion.pme --all-observations --output results/my-inversion
```

The default export writes separate FITS files for the three spherical magnetic components. See [exports](exports.md) for other parameters and formats.

Resume the same run explicitly. To extend it, increase `training.max_steps` in your YAML first:

```bash
spinnme run --config inversion.yaml --data_path ./data/hmi \
  --run_path ./runs/my-inversion --resume
```

Resume requires `last.ckpt` and `resolved.yaml`. Keep the data, model, instrument, loss, normalization, optimizer settings consistent. Use a new output directory when changing the scientific setup. A `.pme` snapshot cannot replace a training checkpoint.

Retain `resolved.yaml`, the manifest, inputs and calibration to repeat a setup. Loading a snapshot reuses its fitted weights. A fresh fit has stochastic initialization and sampling; identical settings do not promise identical maps, nor does resume guarantee bitwise equality across software/hardware. Optimization variability is only one source of uncertainty.

## Weights & Biases

Logging is disabled by default. To record a run locally, install the optional dependency and edit the YAML:

```bash
pip install 'spinnme[wandb]'
```

```yaml
logging:
  mode: offline
  project: spinnme
  name: my-inversion
training:
  check_val_every_n_epoch: null
  val_check_interval: 1000
  plot_validation: true
```

The W&B project is always `spinnme`. Use a short representative `name`, such as `hmi-90s` or `hinode-sp`. `offline` saves W&B records under the working directory. For a dashboard with curves and images, authenticate with `wandb login` and use `mode: online`. Set `mode: disabled` to turn logging off.

From a cloned repository, install the extra into your active environment and authenticate:

```bash
python -m pip install -e '.[wandb]'
wandb login
```

Merge these settings into the existing YAML sections to select your W&B team and update the loss display every 10 training batches:

```yaml
logging:
  mode: online
  project: spinnme
  entity: your-wandb-team
  name: hmi-north-20260110
  group: hmi-poles-20260110
training:
  enable_progress_bar: true
  metric_log_interval: 10
  log_every_n_steps: 10
```

Replace `your-wandb-team` with the entity slug from your W&B project URL, or omit `entity` to use your account's default destination. Use separate run names for north and south and the same group to compare them. Authentication belongs in `wandb login`, not in the YAML. The metric window counts training batches; `log_every_n_steps` controls Lightning's logger cadence in optimizer steps. With the default `accumulate_grad_batches: 1`, the example aligns both intervals. Set both intervals to `1` for per-batch values, at the cost of more logging and device synchronization.

Plots are enabled by default and logged every `val_check_interval` training batches (1000 in the earlier example). They include magnetic/velocity and ME parameter maps, integrated I/Q/U/V comparisons, and representative spectral fits; enabled limb/artifact corrections add their own maps. Images carry the current `trainer/global_step`. Set `plot_validation: false` for scalar metrics only. `data.valid_config.resolution` controls the map size and rendering cost. Full maps require `limit_val_batches: 1.0` (the default); partial validation still reports scalar metrics.

For convergence or memory problems, start with a short run and use the [customization recipes](training_options.md) and [troubleshooting guide](troubleshooting.md).
