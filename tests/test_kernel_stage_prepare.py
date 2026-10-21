import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from kernel_stage_search.cli import load_config
from kernel_stage_search.protocol import (PROJECT_ROOT, bind_preparation, preparation_metadata,
                                          validate_preparation_checkpoint, validate_training_entry)


class PrepareTests(unittest.TestCase):
    def config(self):
        return load_config(PROJECT_ROOT/"configs/kernel_stage_search/prepare_bbbb.yaml")

    def test_prepare_uses_no_official_loader_or_validation_and_saves_endpoint(self):
        module = importlib.import_module("main")
        config = self.config()
        model = torch.nn.Linear(2, 2)
        model.cuda = lambda: model
        loader = [None, None]
        config.defrost()
        config.OUTPUT = tempfile.gettempdir()
        config.freeze()
        with mock.patch.multiple(module, logger=mock.Mock(), build_model=mock.Mock(return_value=model),
                                 build_mtl_model=mock.Mock(return_value=model),
                                 get_model_complexity_info=mock.Mock(return_value=("0", "0")),
                                 build_optimizer=mock.Mock(), NativeScalerWithGradNormCount=mock.Mock(),
                                 build_scheduler=mock.Mock(), get_loss=mock.Mock(side_effect=lambda *a: torch.nn.Identity()),
                                 ConstrainedMTLController=mock.Mock(return_value=None),
                                 load_checkpoint=mock.Mock(), train_one_epoch=mock.Mock(return_value=(None, 0., 0.)),
                                 save_checkpoint=mock.Mock(), build_loader=mock.Mock(), validate=mock.Mock(),
                                 wandb_available=False, create=True), \
                mock.patch("stage_ablation.prepare_trainable_parameters"), \
                mock.patch("kernel_stage_search.protocol.write_json"), \
                mock.patch("kernel_stage_search.data.build_prepare_train_loader", return_value=(loader, None, loader, None, None)), \
                mock.patch.object(module.dist, "is_initialized", return_value=False), \
                mock.patch.object(module.dist, "get_rank", return_value=0):
            module.main(config)
            module.build_loader.assert_not_called()
            module.validate.assert_not_called()
            self.assertEqual(module.train_one_epoch.call_count, 60)
            self.assertEqual(module.save_checkpoint.call_args.args[1], 59)
            self.assertEqual(module.build_scheduler.call_args.args[0].TRAIN.EPOCHS, 300)
            self.assertEqual(module.load_checkpoint.call_args.kwargs["initialization_path"], config.MODEL.INIT_CHECKPOINT)

    def test_preparation_identity_and_resume_barrier(self):
        config = self.config()
        bind_preparation(config, {"protocol_id": "base", "semantic": {"split_id": "split"}})
        middle, final = preparation_metadata(config, 58), preparation_metadata(config, 59)
        self.assertFalse(middle["prepared_complete"])
        self.assertTrue(final["prepared_complete"])
        checkpoint = {"kernel_search_preparation": final, "epoch": 59, "optimizer": {}, "lr_scheduler": {}, "scaler": {}}
        validate_preparation_checkpoint(config, checkpoint)
        with self.assertRaises(ValueError):
            validate_preparation_checkpoint(config, checkpoint, initialization=True)
        formal = config.clone()
        formal.defrost()
        formal.MODEL.KERNEL_SEARCH.ENABLED = False
        formal.MODEL.KERNEL_SEARCH.PHASE = "off"
        formal.freeze()
        with self.assertRaisesRegex(ValueError, "formal"):
            validate_preparation_checkpoint(formal, checkpoint)

    def test_score_training_and_prepare_eval_are_rejected(self):
        score = load_config(PROJECT_ROOT/"configs/kernel_stage_search/swin_t_pascal.yaml")
        with self.assertRaisesRegex(ValueError, "score"):
            validate_training_entry(score)
        config = self.config()
        config.defrost()
        config.EVAL_MODE = True
        config.freeze()
        with self.assertRaisesRegex(ValueError, "eval"):
            validate_training_entry(config)

    def test_lr_scaling_is_once_and_keeps_horizon(self):
        from utils import scale_learning_rates
        config = self.config()
        scale_learning_rates(config, 1)
        self.assertAlmostEqual(config.TRAIN.BASE_LR, 1.953125e-5)
        self.assertEqual(config.TRAIN.EPOCHS, 300)
        with self.assertRaises(RuntimeError):
            scale_learning_rates(config, 1)
        from kernel_stage_search.protocol import resolved_identity, search_settings
        score = load_config(PROJECT_ROOT/"configs/kernel_stage_search/swin_t_pascal.yaml")
        self.assertEqual(resolved_identity(score), resolved_identity(config))
        self.assertEqual(search_settings(score), search_settings(config))

    def test_prepared_registration_rejects_incomplete_or_changed_checkpoint(self):
        from kernel_stage_search.checkpoint import register_prepared_checkpoint, verify_prepared
        from stage_ablation import stage_ablation_metadata
        config = self.config()
        bind_preparation(config, {"protocol_id": "base", "semantic": {"split_id": "split"}})
        checkpoint = {"model": torch.nn.Linear(2, 2).state_dict(), "epoch": 59,
                      "kernel_search_preparation": preparation_metadata(config, 59),
                      "stage_ablation": stage_ablation_metadata(config), "optimizer": {}, "lr_scheduler": {}, "scaler": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mb.pth"
            torch.save(checkpoint, path)
            manifest = register_prepared_checkpoint(config, path)
            verify_prepared(config, path, manifest)
            checkpoint["epoch"] = 58
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_prepared(config, path, manifest)
            with self.assertRaisesRegex(ValueError, "endpoint"):
                register_prepared_checkpoint(config, path)
