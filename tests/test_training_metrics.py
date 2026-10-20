from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _training_helpers import _module, _physical_batch

from spinnme.train.metrics import TrainingMetricWindow


def test_window_detaches_and_averages_in_one_collective(monkeypatch):
    import spinnme.train.metrics as metrics_module
    window = TrainingMetricWindow()
    leaf = torch.tensor(2., requires_grad=True)
    window.add({'loss': leaf.square(), 'weights.I': torch.tensor(1.)})
    window.add({'weights.I': torch.tensor(1.), 'loss': leaf * 4})
    assert window.total.grad_fn is None
    monkeypatch.setattr(metrics_module.dist, 'is_initialized', lambda: True)
    calls = []
    def reduce(packed, op):
        calls.append(packed.clone())
        # A second rank contributes two batches, losses 10 and 14.
        packed.add_(torch.tensor([24., 2., 2.]))
    monkeypatch.setattr(metrics_module.dist, 'all_reduce', reduce)
    result = window.drain()
    assert len(calls) == 1
    assert result['loss'] == 9
    assert result['weights.I'] == 1
    assert window.count == 0
    assert window.drain() == {}


def test_module_logs_only_at_interval_and_flushes_short_epoch(monkeypatch):
    module = _module(None)
    module.metric_log_interval = 2
    monkeypatch.setattr(module, 'optimizers', lambda: SimpleNamespace(param_groups=[{'lr': .001}]))
    logger = Mock()
    loss_logger = Mock()
    monkeypatch.setattr(module, 'log_dict', logger)
    monkeypatch.setattr(module, 'log', loss_logger)
    for i in range(3):
        module.on_train_batch_end({'loss': torch.tensor(float(i + 1)), 'weights.I': torch.tensor(1.)}, {}, i)
        assert logger.call_count == (i + 1) // 2
        assert loss_logger.call_count == (i + 1) // 2
    assert loss_logger.call_args.args[0] == 'train.loss'
    assert loss_logger.call_args.args[1] == 1.5
    assert logger.call_args.kwargs['sync_dist'] is False
    module.on_train_epoch_end()
    assert logger.call_count == 2
    assert loss_logger.call_args.args[1] == 3
    assert logger.call_args.kwargs['on_epoch'] is True
    module.on_train_epoch_end()
    assert logger.call_count == 2


def test_time_shift_decay_and_resume_do_not_read_device_scalars_per_batch(monkeypatch):
    module = _module(None)
    module._time_shift_enabled = True
    module.time_shift_scale.fill_(.2)
    module.time_shift_scale_gamma = .15
    monkeypatch.setattr(module, 'optimizers', lambda: SimpleNamespace(param_groups=[{'lr': .001}]))
    module.on_train_batch_end({'loss': torch.tensor(1.)}, {}, 0)
    torch.testing.assert_close(module.time_shift_scale, torch.tensor(.05))
    module.on_train_batch_end({'loss': torch.tensor(1.)}, {}, 1)
    assert module.time_shift_scale == 0
    checkpoint = {'state_dict': module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    assert module._train_metrics.count == 0
    assert not module._time_shift_enabled


@pytest.mark.parametrize('logging_enabled', [False, True])
def test_lightning_training_logs_windows_and_final_partial_window(tmp_path, logging_enabled):
    import csv

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    module = _module(None)
    module.metric_log_interval = 2
    logger = CSVLogger(tmp_path) if logging_enabled else False
    trainer = Trainer(accelerator='cpu', devices=1, max_epochs=1, logger=logger,
                      log_every_n_steps=1,
                      enable_checkpointing=False, enable_model_summary=False,
                      enable_progress_bar=False, num_sanity_val_steps=0)
    loader = DataLoader([_physical_batch() for _ in range(3)], batch_size=None)
    trainer.fit(module, train_dataloaders=loader)
    assert trainer.global_step == 3
    assert module._train_metrics.count == 0
    assert torch.isfinite(trainer.callback_metrics['train.loss'])
    assert set(trainer.progress_bar_metrics) == {'train.loss'}
    assert trainer.progress_bar_metrics['train.loss'] == trainer.callback_metrics['train.loss']
    assert 'train.replay_fraction' not in trainer.callback_metrics
    if logging_enabled:
        with open(logger.experiment.metrics_file_path) as stream:
            rows = list(csv.DictReader(stream))
        losses = [float(row['train.loss']) for row in rows if row.get('train.loss')]
        assert len(losses) >= 2
        assert losses[-1] == trainer.progress_bar_metrics['train.loss']


def test_physics_metric_changes_flush_previous_window(monkeypatch):
    module = _module(None)
    monkeypatch.setattr(module, 'optimizers', lambda: SimpleNamespace(param_groups=[{'lr': .001}]))
    logger = Mock()
    loss_logger = Mock()
    monkeypatch.setattr(module, 'log_dict', logger)
    monkeypatch.setattr(module, 'log', loss_logger)
    module.on_train_batch_end({'loss': torch.tensor(1.)}, {}, 0)
    module.on_train_batch_end({'loss': torch.tensor(2.), 'physics.divergence': torch.tensor(.2)}, {}, 1)
    assert logger.call_count == 1
    assert loss_logger.call_args.args[1] == 1
    assert module._train_metrics.count == 1
    module.on_train_epoch_end()
    assert logger.call_args.args[0]['train.physics.divergence'] == .2
