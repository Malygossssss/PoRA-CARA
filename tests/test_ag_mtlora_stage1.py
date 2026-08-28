import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import yaml
from yacs.config import CfgNode as CN

import ag_mtlora.stage1 as stage1
import config as config_module
import models.swin_transformer_mtlora as swin_mtlora_module
from mtl_loss_schemes import SoftMaxwithLoss
from utils import scale_learning_rates


class FakeDataset:
    def __init__(self, sample_ids):
        self.im_ids = list(sample_ids)

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, index):
        return {"image": index}


class DummyResolvedConfig:
    def dump(self):
        return "MODEL:\n  AGMTLORA:\n    ENABLED: true\n"


class FakePromptBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.ones(1))
        self.deep_prompt_embeddings = nn.Parameter(torch.ones(1))
        self.lora_shared_A = nn.Parameter(torch.ones(1))
        self.non_lora_weight = nn.Parameter(torch.ones(1))


class FakePromptTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakePromptBackbone()
        self.decoder = nn.Linear(1, 1)


class SmokeTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.head = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, image):
        return {"task_a": self.head(self.norm(image))}


class SmokeTestCriterion(nn.Module):
    def forward(self, outputs, targets):
        task_loss = (outputs["task_a"] - targets["task_a"]).square().mean()
        return task_loss, {"task_a": task_loss, "total": task_loss}


class FakeStage1LossScaler:
    def __init__(self, scale_before, scale_after, grad_norm):
        self.scale = float(scale_before)
        self.scale_after = float(scale_after)
        self.grad_norm = float(grad_norm)

    def state_dict(self):
        return {"scale": self.scale}

    def __call__(self, *args, **kwargs):
        self.scale = self.scale_after
        return torch.tensor(self.grad_norm)


class FakeStage1Scheduler:
    def __init__(self):
        self.updates = []

    def step_update(self, update):
        self.updates.append(update)

    def state_dict(self):
        return {"updates": list(self.updates)}


def build_stage1_config(tmpdir, split_mode="train_meta_strict"):
    config = CN()
    config.SEED = 7
    config.TASKS = ["task_a", "task_b"]

    config.DATA = CN()
    config.DATA.DBNAME = "NYUD"
    config.DATA.DATA_PATH = tmpdir
    config.DATA.BATCH_SIZE = 2
    config.DATA.NUM_WORKERS = 0
    config.DATA.PIN_MEMORY = False

    config.MODEL = CN()
    config.MODEL.SWIN = CN()
    config.MODEL.SWIN.DEPTHS = [2]
    config.MODEL.AGMTLORA = CN(new_allowed=True)
    config.MODEL.AGMTLORA.DATA_SPLIT_MODE = split_mode
    config.MODEL.AGMTLORA.META_VAL_RATIO = 0.2
    config.MODEL.AGMTLORA.RESOLVED_META_SPLIT_SEED = 13
    config.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH = os.path.join(tmpdir, "meta_split.json")
    config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH = os.path.join(tmpdir, "affinity.json")
    config.MODEL.AGMTLORA.GROUPING_SAVE_PATH = os.path.join(tmpdir, "grouping.json")
    config.MODEL.AGMTLORA.VISUALIZE_SYMMETRIC_AFFINITY = False
    config.MODEL.AGMTLORA.PARTITION_GRANULARITY = "global"
    config.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE = "final_predictions"
    config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_BUDGET = 3
    config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_STRATEGY = "all_singletons+all_pairs+random_higher_order"
    config.MODEL.AGMTLORA.BASE_PREDICTOR = "spline_ridge"
    config.MODEL.AGMTLORA.BASE_PREDICTOR_KWARGS = CN(new_allowed=True)
    config.MODEL.AGMTLORA.RESIDUAL_PREDICTOR = "ridge"
    config.MODEL.AGMTLORA.RESIDUAL_ALPHA = 1.0
    config.MODEL.AGMTLORA.MAX_GROUPS = 2
    config.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET = 2
    config.MODEL.AGMTLORA.GROUP_SHARED_RANKS = []
    config.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION = "equal_split"
    config.MODEL.AGMTLORA.SEARCH_OBJECTIVE = "mean_final_predicted_gain"

    config.TASKS_CONFIG = CN(new_allowed=True)
    config.TASKS_CONFIG.ALL_TASKS = CN(new_allowed=True)
    config.TASKS_CONFIG.ALL_TASKS.FLAGVALS = CN(new_allowed=True)
    config.TASKS_CONFIG.FLAGVALS = CN(new_allowed=True)

    config.OUTPUT = tmpdir
    config.freeze()
    return config


def build_update_config_args(cfg_path, tmpdir, tasks="task_a,task_b"):
    return SimpleNamespace(
        cfg=cfg_path,
        opts=None,
        local_rank=0,
        tasks=tasks,
        nyud=tmpdir,
        output=tmpdir,
        tag="unittest",
        name="test_model",
    )


class Stage1MetaSplitTest(unittest.TestCase):
    def test_amp_scale_backoff_is_treated_as_recoverable_skipped_step(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        scheduler = FakeStage1Scheduler()
        runtime = stage1.Stage1TrainRuntime(
            optimizer=optimizer,
            lr_scheduler=scheduler,
            loss_scaler=FakeStage1LossScaler(65536.0, 32768.0, float("inf")),
            amp_enabled=True,
            clip_grad=5.0,
        )

        result = stage1.stage1_optimizer_step(
            runtime,
            torch.tensor(1.0, requires_grad=True),
            model,
            {"phase": "warmup", "epoch": 0, "batch": 0},
        )

        self.assertTrue(result.optimizer_step_skipped)
        self.assertIsNone(result.grad_norm)
        self.assertEqual(result.loss_scale_before, 65536.0)
        self.assertEqual(result.loss_scale_after, 32768.0)
        self.assertEqual(runtime.amp_overflow_count, 1)
        self.assertEqual(runtime.consecutive_amp_overflows, 1)
        self.assertEqual(runtime.global_step, 1)
        self.assertEqual(scheduler.updates, [0])

    def test_nonfinite_grad_without_amp_scale_backoff_still_aborts(self):
        model = nn.Linear(2, 1)
        runtime = stage1.Stage1TrainRuntime(
            optimizer=torch.optim.SGD(model.parameters(), lr=1e-4),
            lr_scheduler=FakeStage1Scheduler(),
            loss_scaler=FakeStage1LossScaler(32768.0, 32768.0, float("inf")),
            amp_enabled=True,
            clip_grad=5.0,
        )

        with self.assertRaisesRegex(stage1.Stage1NumericalError, "grad_norm"):
            stage1.stage1_optimizer_step(
                runtime,
                torch.tensor(1.0, requires_grad=True),
                model,
                {"phase": "warmup", "epoch": 0, "batch": 1},
            )

    def test_nonfinite_guard_reports_tensor_and_training_context(self):
        context = {"phase": "warmup", "epoch": 2, "batch": 17, "task": "normals"}

        with self.assertRaises(stage1.Stage1NumericalError) as raised:
            stage1.ensure_finite_tensor(
                torch.tensor([1.0, float("nan")]),
                "task_losses.normals",
                context,
            )

        self.assertEqual(raised.exception.details["phase"], "warmup")
        self.assertEqual(raised.exception.details["epoch"], 2)
        self.assertEqual(raised.exception.details["batch"], 17)
        self.assertEqual(raised.exception.details["tensor_name"], "task_losses.normals")
        self.assertEqual(raised.exception.details["tensor_stats"]["nonfinite_count"], 1)

    def test_amp_smoke_test_restores_batchnorm_state_and_mode(self):
        model = SmokeTestModel()
        model.eval()
        state_before = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        batch = {
            "image": torch.randn(2, 1, 4, 4),
            "task_a": torch.randn(2, 1, 4, 4),
            "meta": {"image": ["sample_a", "sample_b"]},
        }

        result = stage1.run_stage1_amp_smoke_test(
            model,
            batch,
            SmokeTestCriterion(),
            ["task_a"],
            torch.device("cpu"),
            amp_enabled=False,
        )

        self.assertFalse(model.training)
        self.assertGreater(result["checked_gradients"], 0)
        self.assertEqual(result["sample_ids"], ["sample_a", "sample_b"])
        for name, tensor_before in state_before.items():
            self.assertTrue(torch.equal(model.state_dict()[name], tensor_before), name)

    def test_scale_learning_rates_matches_formal_training_and_rejects_double_scaling(self):
        config = CN()
        config.DATA = CN()
        config.DATA.BATCH_SIZE = 9
        config.TRAIN = CN()
        config.TRAIN.BASE_LR = 1.0e-3
        config.TRAIN.WARMUP_LR = 1.0e-6
        config.TRAIN.MIN_LR = 1.0e-5
        config.TRAIN.ACCUMULATION_STEPS = 1
        config.TRAIN.LR_SCALED = False
        config.freeze()

        scaled = scale_learning_rates(config, world_size=1)

        self.assertAlmostEqual(config.TRAIN.BASE_LR, 1.7578125e-5)
        self.assertAlmostEqual(config.TRAIN.WARMUP_LR, 1.7578125e-8)
        self.assertAlmostEqual(config.TRAIN.MIN_LR, 1.7578125e-7)
        self.assertEqual(scaled["global_batch_size"], 9)
        self.assertTrue(config.TRAIN.LR_SCALED)
        with self.assertRaisesRegex(RuntimeError, "already been scaled"):
            scale_learning_rates(config, world_size=1)

    def test_softmax_loss_returns_differentiable_zero_for_all_ignored_batch(self):
        output = torch.randn(2, 7, 4, 4, requires_grad=True)
        target = torch.full((2, 1, 4, 4), 255, dtype=torch.long)

        loss = SoftMaxwithLoss(ignore_index=255)(output, target)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(output.grad)
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertEqual(int(torch.count_nonzero(output.grad).item()), 0)

    def test_stage1_scheduler_uses_search_duration_and_affinity_warmup(self):
        config = CN()
        config.TRAIN = CN()
        config.TRAIN.EPOCHS = 300
        config.TRAIN.WARMUP_EPOCHS = 20
        config.freeze()

        scheduler_config = stage1.build_stage1_scheduler_config(
            config,
            warmup_epochs=5,
            total_epochs=55,
        )

        self.assertEqual(scheduler_config.TRAIN.EPOCHS, 55)
        self.assertEqual(scheduler_config.TRAIN.WARMUP_EPOCHS, 5)
        self.assertEqual(config.TRAIN.EPOCHS, 300)
        self.assertEqual(config.TRAIN.WARMUP_EPOCHS, 20)

    def test_build_task_model_keeps_prompt_trainable_for_stage1_optimizer(self):
        config = CN()
        config.MTL = False
        config.MODEL = CN()
        config.MODEL.MTLORA = CN()
        config.MODEL.MTLORA.ENABLED = True
        config.MODEL.MTLORA.FREEZE_PRETRAINED = True
        config.MODEL.MTLORA.BIAS = "none"
        config.MODEL.MTLORA.DOWNSAMPLER_ENABLED = False
        config.MODEL.PROMPT = CN()
        config.MODEL.PROMPT.ENABLED = True
        config.TRAIN = CN()
        config.TRAIN.FREEZE_PATCH_EMBED = True
        config.TRAIN.FREEZE_LAYER_NORM = True
        config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS = True
        config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION = True
        config.TRAIN.BASE_LR = 1e-4
        config.TRAIN.WEIGHT_DECAY = 0.05
        config.TRAIN.OPTIMIZER = CN()
        config.TRAIN.OPTIMIZER.NAME = "adamw"
        config.TRAIN.OPTIMIZER.EPS = 1e-8
        config.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
        config.TRAIN.OPTIMIZER.MOMENTUM = 0.9
        config.freeze()

        with mock.patch("ag_mtlora.stage1.build_model", return_value=FakePromptTaskModel()), mock.patch(
            "ag_mtlora.stage1.maybe_load_initial_weights"
        ):
            model = stage1.build_task_model(config, torch.device("cpu"), mock.Mock())

        prompt_params = [
            param
            for name, param in model.backbone.named_parameters()
            if "prompt_embeddings" in name or "deep_prompt_embeddings" in name
        ]
        self.assertEqual(len(prompt_params), 2)
        self.assertTrue(all(param.requires_grad for param in prompt_params))
        self.assertFalse(model.backbone.non_lora_weight.requires_grad)

        optimizer = stage1.build_optimizer(config, model)
        optimizer_param_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        self.assertTrue(all(id(param) in optimizer_param_ids for param in prompt_params))

    def test_parse_predictor_progress_from_log_recovers_singletons_and_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "log_rank0.txt")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("Predictor target collected | singleton task=semseg | val_loss=1.234500\n")
                handle.write("Predictor target collected | group=['semseg', 'normals'] | gains={'semseg': 0.1, 'normals': 0.2}\n")

            predictor_targets, singleton_losses = stage1.parse_predictor_progress_from_log(log_path)

            self.assertEqual(singleton_losses, {"semseg": 1.2345})
            self.assertEqual(predictor_targets["semseg"], {"semseg": 0.0})
            self.assertEqual(predictor_targets["semseg|normals"], {"semseg": 0.1, "normals": 0.2})

    def test_run_partition_search_uses_generic_search_scores_with_existing_tie_breaks(self):
        tasks = ["task_a", "task_b", "task_c"]
        search_scores = {}
        for group in stage1.enumerate_candidate_groups(tasks):
            search_scores[stage1.group_to_key(group)] = {task: 0.0 for task in group}

        ranked_partitions = stage1.run_partition_search(tasks, search_scores, max_groups=2)

        self.assertEqual(
            [item["groups"] for item in ranked_partitions],
            [
                [["task_a", "task_b", "task_c"]],
                [["task_a"], ["task_b", "task_c"]],
                [["task_a", "task_b"], ["task_c"]],
                [["task_a", "task_c"], ["task_b"]],
            ],
        )

    def test_build_stage1_data_split_manifest_is_deterministic_and_persists_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            sample_ids = [f"sample_{idx}" for idx in range(10)]

            with mock.patch(
                "ag_mtlora.stage1.get_mtl_split_sample_ids",
                return_value=sample_ids,
            ):
                logger = mock.Mock()
                manifest_a = stage1.build_stage1_data_split_manifest(config, logger)
                manifest_b = stage1.build_stage1_data_split_manifest(config, logger)

            self.assertEqual(manifest_a["meta_train_indices"], manifest_b["meta_train_indices"])
            self.assertEqual(manifest_a["meta_val_indices"], manifest_b["meta_val_indices"])
            self.assertEqual(len(manifest_a["meta_val_indices"]), 2)
            self.assertEqual(len(manifest_a["meta_train_indices"]), 8)
            self.assertEqual(
                sorted(manifest_a["meta_train_indices"] + manifest_a["meta_val_indices"]),
                list(range(10)),
            )
            self.assertTrue(set(manifest_a["meta_train_indices"]).isdisjoint(manifest_a["meta_val_indices"]))
            self.assertTrue(os.path.exists(manifest_a["meta_split_path"]))

            with open(manifest_a["meta_split_path"], "r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["meta_train_ids"], manifest_a["meta_train_ids"])
            self.assertEqual(persisted["meta_val_ids"], manifest_a["meta_val_ids"])

    def test_build_stage1_data_loaders_strict_uses_train_split_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            sample_ids = [f"sample_{idx}" for idx in range(6)]
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "meta_train_indices": [0, 2, 4, 5],
                "meta_val_indices": [1, 3],
                "meta_train_ids": [sample_ids[idx] for idx in [0, 2, 4, 5]],
                "meta_val_ids": [sample_ids[idx] for idx in [1, 3]],
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }

            with mock.patch("ag_mtlora.stage1.get_transformations", return_value=("train_tf", "eval_tf")), mock.patch(
                "ag_mtlora.stage1.get_mtl_dataset",
                side_effect=lambda db_name, cfg, transforms, split: FakeDataset(sample_ids),
            ) as mocked_get_dataset, mock.patch(
                "ag_mtlora.stage1.get_mtl_train_dataloader",
                side_effect=lambda cfg, dataset: ("train_loader", len(dataset)),
            ) as mocked_train_loader, mock.patch(
                "ag_mtlora.stage1.get_mtl_val_dataloader",
                side_effect=lambda cfg, dataset: ("val_loader", len(dataset)),
            ) as mocked_val_loader:
                dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = stage1.build_stage1_data_loaders(
                    config,
                    data_split_manifest=manifest,
                )

            self.assertEqual(len(dataset_train), 4)
            self.assertEqual(len(dataset_val), 2)
            self.assertEqual(data_loader_train, ("train_loader", 4))
            self.assertEqual(data_loader_val, ("val_loader", 2))
            self.assertIsNone(mixup_fn)
            self.assertEqual(mocked_get_dataset.call_count, 2)
            self.assertTrue(all(call.kwargs["split"] == "train" for call in mocked_get_dataset.call_args_list))
            mocked_train_loader.assert_called_once()
            mocked_val_loader.assert_called_once()

    def test_build_stage1_data_loaders_projects_manifest_ids_for_filtered_task_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "meta_train_indices": [0, 2, 4, 5],
                "meta_val_indices": [1, 3],
                "meta_train_ids": ["sample_0", "sample_2", "sample_4", "sample_5"],
                "meta_val_ids": ["sample_1", "sample_3"],
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }
            filtered_sample_ids = ["sample_1", "sample_2", "sample_5"]

            with mock.patch("ag_mtlora.stage1.get_transformations", return_value=("train_tf", "eval_tf")), mock.patch(
                "ag_mtlora.stage1.get_mtl_dataset",
                side_effect=lambda db_name, cfg, transforms, split: FakeDataset(filtered_sample_ids),
            ), mock.patch(
                "ag_mtlora.stage1.get_mtl_train_dataloader",
                side_effect=lambda cfg, dataset: ("train_loader", list(dataset.indices)),
            ), mock.patch(
                "ag_mtlora.stage1.get_mtl_val_dataloader",
                side_effect=lambda cfg, dataset: ("val_loader", list(dataset.indices)),
            ):
                dataset_train, dataset_val, data_loader_train, data_loader_val, _ = stage1.build_stage1_data_loaders(
                    config,
                    data_split_manifest=manifest,
                )

            self.assertEqual(list(dataset_train.indices), [1, 2])
            self.assertEqual(list(dataset_val.indices), [0])
            self.assertEqual(data_loader_train, ("train_loader", [1, 2]))
            self.assertEqual(data_loader_val, ("val_loader", [0]))

    def test_create_resolved_training_config_is_schema_safe_and_base_backed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg_path = os.path.join(tmpdir, "base.yaml")
            grouping_json_path = os.path.join(tmpdir, "grouping.json")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                handle.write("MODEL:\n  NAME: test\n")

            resolved = stage1.create_resolved_training_config(
                base_cfg_path,
                grouping_json_path,
                [[32, 32], [16, 16]],
            )

            payload = yaml.safe_load(resolved.dump())
            self.assertEqual(payload["BASE"], [os.path.abspath(base_cfg_path)])
            self.assertEqual(payload["MODEL"]["AGMTLORA"]["GROUPING_SOURCE"], "fixed_json")
            self.assertEqual(payload["MODEL"]["AGMTLORA"]["GROUPING_JSON"], os.path.abspath(grouping_json_path))
            self.assertEqual(payload["MODEL"]["AGMTLORA"]["GROUP_SHARED_RANKS"], [[32, 32], [16, 16]])

    def test_update_config_loads_stage_wise_grouping_json_into_runtime_routing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            grouping_json_path = os.path.join(tmpdir, "grouping.json")
            with open(grouping_json_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": ["task_a", "task_b"],
                        "partition_granularity": "stage",
                        "group_slot_names": ["group_0", "group_1"],
                        "groups_by_stage": [
                            [["task_a", "task_b"]],
                            [["task_a"], ["task_b"]],
                        ],
                        "group_shared_ranks": [[4, 2], [0, 2]],
                    },
                    handle,
                    indent=2,
                )

            cfg_path = os.path.join(tmpdir, "config.yaml")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "BASE": [""],
                        "MODEL": {
                            "MTLORA": {"ENABLED": True},
                            "AGMTLORA": {
                                "ENABLED": True,
                                "GROUPING_SOURCE": "fixed_json",
                                "GROUPING_JSON": grouping_json_path,
                                "PARTITION_GRANULARITY": "stage",
                            },
                            "SWIN": {"DEPTHS": [2, 2]},
                        },
                    },
                    handle,
                    sort_keys=False,
                )

            dummy_task_cfg = {
                "ALL_TASKS": {
                    "NUM_OUTPUT": {},
                    "FLAGVALS": {},
                    "INFER_FLAGVALS": {},
                }
            }
            config = config_module._C.clone()
            with mock.patch("config.get_tasks_config", return_value=(dummy_task_cfg, None)):
                config_module.update_config(config, build_update_config_args(cfg_path, tmpdir))

            self.assertEqual(config.MODEL.AGMTLORA.RESOLVED_PARTITION_GRANULARITY, "stage")
            self.assertEqual(config.MODEL.MTLORA.AGMTLORA_GROUP_NAMES, ["group_0", "group_1"])
            self.assertEqual(config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS, [[4, 2], [0, 2]])
            self.assertEqual(
                dict(config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE["stage_0"]),
                {"task_a": "group_0", "task_b": "group_0"},
            )
            self.assertEqual(
                dict(config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE["stage_1"]),
                {"task_a": "group_0", "task_b": "group_1"},
            )

    def test_model_task_routing_prefers_stage_specific_mapping(self):
        mtlora = SimpleNamespace(
            AGMTLORA_ENABLED=True,
            AGMTLORA_TASK_TO_GROUP={"task_a": "group_0", "task_b": "group_0"},
            AGMTLORA_TASK_TO_GROUP_BY_STAGE={
                "stage_0": {"task_a": "group_0", "task_b": "group_0"},
                "stage_1": {"task_a": "group_1", "task_b": "group_0"},
            },
        )

        self.assertEqual(
            swin_mtlora_module._get_task_to_group(mtlora, 0),
            {"task_a": "group_0", "task_b": "group_0"},
        )
        self.assertEqual(
            swin_mtlora_module._get_task_to_group(mtlora, 1),
            {"task_a": "group_1", "task_b": "group_0"},
        )

    def test_build_stage1_data_loaders_legacy_mode_uses_existing_build_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir, split_mode="official_val")
            expected = ("dataset_train", "dataset_val", "loader_train", "loader_val", None)

            with mock.patch("ag_mtlora.stage1.build_loader", return_value=expected) as mocked_build_loader:
                result = stage1.build_stage1_data_loaders(
                    config,
                    data_split_manifest={"selection_data_split_mode": "official_val"},
                )

            self.assertEqual(result, expected)
            mocked_build_loader.assert_called_once_with(config)

    def test_collect_predictor_training_targets_reuses_same_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = CN()
            base_config.TASKS = ["task_a", "task_b"]
            manifest = {"selection_data_split_mode": "train_meta_strict"}

            def make_group_config(group):
                cfg = CN()
                cfg.TASKS = list(group)
                cfg.MODEL = CN()
                cfg.MODEL.AGMTLORA = CN()
                cfg.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS = 1
                cfg.OUTPUT = tmpdir
                return cfg

            short_train_side_effects = [
                (None, {"task_a": 1.0}),
                (None, {"task_b": 2.0}),
                (None, {"task_a": 0.7, "task_b": 1.6}),
            ]

            with mock.patch(
                "ag_mtlora.stage1.rebuild_task_config",
                side_effect=lambda config, group, output_dir=None: make_group_config(group),
            ), mock.patch(
                "ag_mtlora.stage1.short_train_model",
                side_effect=short_train_side_effects,
            ) as mocked_short_train:
                predictor_targets, singleton_losses = stage1.collect_predictor_training_targets(
                    base_config,
                    mock.Mock(),
                    baseline_state_dict={},
                    train_groups=[["task_a"], ["task_b"], ["task_a", "task_b"]],
                    working_dir=tmpdir,
                    data_split_manifest=manifest,
                )

            self.assertEqual(singleton_losses, {"task_a": 1.0, "task_b": 2.0})
            self.assertEqual(predictor_targets["task_a"], {"task_a": 0.0})
            self.assertEqual(predictor_targets["task_b"], {"task_b": 0.0})
            self.assertAlmostEqual(predictor_targets["task_a|task_b"]["task_a"], 0.3)
            self.assertAlmostEqual(predictor_targets["task_a|task_b"]["task_b"], 0.4)
            self.assertEqual(mocked_short_train.call_count, 3)
            self.assertTrue(all(call.kwargs["data_split_manifest"] is manifest for call in mocked_short_train.call_args_list))

    def test_collect_predictor_training_targets_uses_cached_log_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "log_rank0.txt")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("Predictor target collected | singleton task=task_a | val_loss=1.000000\n")

            base_config = CN()
            base_config.TASKS = ["task_a", "task_b"]
            manifest = {"selection_data_split_mode": "train_meta_strict", "meta_split_path": os.path.join(tmpdir, "meta_split.json")}

            def make_group_config(group):
                cfg = CN()
                cfg.TASKS = list(group)
                cfg.MODEL = CN()
                cfg.MODEL.AGMTLORA = CN()
                cfg.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS = 1
                cfg.OUTPUT = tmpdir
                return cfg

            with mock.patch(
                "ag_mtlora.stage1.rebuild_task_config",
                side_effect=lambda config, group, output_dir=None: make_group_config(group),
            ), mock.patch(
                "ag_mtlora.stage1.short_train_model",
                side_effect=[
                    (None, {"task_b": 2.0}),
                    (None, {"task_a": 0.8, "task_b": 1.5}),
                ],
            ) as mocked_short_train:
                predictor_targets, singleton_losses = stage1.collect_predictor_training_targets(
                    base_config,
                    mock.Mock(),
                    baseline_state_dict={},
                    train_groups=[["task_a"], ["task_b"], ["task_a", "task_b"]],
                    working_dir=tmpdir,
                    data_split_manifest=manifest,
                )

            self.assertEqual(singleton_losses["task_a"], 1.0)
            self.assertEqual(singleton_losses["task_b"], 2.0)
            self.assertEqual(mocked_short_train.call_count, 2)
            self.assertEqual(predictor_targets["task_a"], {"task_a": 0.0})
            self.assertEqual(predictor_targets["task_b"], {"task_b": 0.0})

    def test_run_stage1_pipeline_records_split_metadata_and_returns_meta_split_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            logger = mock.Mock()
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "selection_eval_label": "meta-val",
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }
            directed_affinity = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
            final_predictions = {
                "task_a": {"task_a": 0.0},
                "task_b": {"task_b": 0.0},
                "task_a|task_b": {"task_a": 0.4, "task_b": 0.6},
            }

            with mock.patch(
                "ag_mtlora.stage1.build_stage1_data_split_manifest",
                return_value=manifest,
            ) as mocked_manifest, mock.patch(
                "ag_mtlora.stage1.warmup_and_collect_affinity",
                return_value={
                    "directed_affinity": directed_affinity,
                    "symmetric_affinity": 0.5 * (directed_affinity + directed_affinity.T),
                    "num_affinity_epochs": 1,
                    "num_batches_per_epoch": [3],
                    "affinity_epoch_history": [directed_affinity],
                    "warmup_checkpoint_path": os.path.join(tmpdir, "warmup_checkpoint.pth"),
                    "post_affinity_checkpoint_path": os.path.join(tmpdir, "post_affinity_checkpoint.pth"),
                    "warmup_validation_losses": {"task_a": 1.0, "task_b": 2.0},
                    "post_affinity_validation_losses": {"task_a": 0.8, "task_b": 1.7},
                    "post_affinity_state_dict": {},
                },
            ) as mocked_warmup, mock.patch(
                "ag_mtlora.stage1.enumerate_candidate_groups",
                return_value=[["task_a"], ["task_b"], ["task_a", "task_b"]],
            ), mock.patch(
                "ag_mtlora.stage1.build_group_proxy",
                return_value=(
                    {
                        "task_a": {"task_a": 0.0},
                        "task_b": {"task_b": 0.0},
                        "task_a|task_b": {"task_a": 1.0, "task_b": 2.0},
                    },
                    [],
                ),
            ), mock.patch(
                "ag_mtlora.stage1.select_predictor_train_groups",
                return_value=[["task_a"], ["task_b"], ["task_a", "task_b"]],
            ), mock.patch(
                "ag_mtlora.stage1.collect_predictor_training_targets",
                return_value=(
                    {
                        "task_a": {"task_a": 0.0},
                        "task_b": {"task_b": 0.0},
                        "task_a|task_b": {"task_a": 0.2, "task_b": 0.3},
                    },
                    {"task_a": 1.0, "task_b": 1.5},
                ),
            ) as mocked_collect, mock.patch(
                "ag_mtlora.stage1.fit_base_predictor",
                return_value={
                    "predictor_name": "mock",
                    "num_samples": 4,
                    "affine_a": 1.0,
                    "affine_b": 0.0,
                },
            ), mock.patch(
                "ag_mtlora.stage1.predict_base_gain",
                side_effect=lambda predictor, proxy_value: float(proxy_value),
            ), mock.patch(
                "ag_mtlora.stage1.fit_residual_predictor",
                return_value=object(),
            ), mock.patch(
                "ag_mtlora.stage1.predict_all_candidate_groups",
                return_value=(
                    final_predictions,
                    final_predictions,
                    final_predictions,
                ),
            ), mock.patch(
                "ag_mtlora.stage1.run_partition_search",
                return_value=[
                    {
                        "groups": [["task_a", "task_b"]],
                        "partition_score": 0.5,
                        "per_task_scores": {"task_a": 0.4, "task_b": 0.6},
                        "num_groups": 1,
                    }
                ],
            ), mock.patch(
                "ag_mtlora.stage1.resolve_group_shared_ranks",
                return_value=([[2]], "auto_equal_split"),
            ), mock.patch(
                "ag_mtlora.stage1.build_task_to_group",
                return_value={"task_a": "group_0", "task_b": "group_0"},
            ), mock.patch(
                "ag_mtlora.stage1.create_resolved_training_config",
                return_value=DummyResolvedConfig(),
            ), mock.patch(
                "ag_mtlora.stage1.save_matrix_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_affinity_epoch_history_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_group_task_rows"
            ), mock.patch(
                "ag_mtlora.stage1.save_predictor_value_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_partition_csv"
            ):
                artifacts = stage1.run_stage1_pipeline(
                    config,
                    tmpdir,
                    logger,
                    base_cfg_path=os.path.join(tmpdir, "base_config.yaml"),
                )

            self.assertEqual(artifacts["meta_split_path"], manifest["meta_split_path"])
            self.assertTrue(os.path.exists(artifacts["resolved_runtime_snapshot_path"]))
            mocked_manifest.assert_called_once_with(config, logger)
            mocked_warmup.assert_called_once()
            self.assertIs(mocked_warmup.call_args.kwargs["data_split_manifest"], manifest)
            self.assertIs(mocked_collect.call_args.kwargs["data_split_manifest"], manifest)

            with open(config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH, "r", encoding="utf-8") as handle:
                affinity_payload = json.load(handle)
            with open(os.path.join(tmpdir, "predictor_train_groups.json"), "r", encoding="utf-8") as handle:
                predictor_payload = json.load(handle)
            with open(config.MODEL.AGMTLORA.GROUPING_SAVE_PATH, "r", encoding="utf-8") as handle:
                grouping_payload = json.load(handle)
            with open(os.path.join(tmpdir, "partition_search_results.json"), "r", encoding="utf-8") as handle:
                partition_payload = json.load(handle)

            for payload in (affinity_payload, predictor_payload, grouping_payload, partition_payload):
                self.assertEqual(payload["selection_data_split_mode"], "train_meta_strict")
                self.assertEqual(payload["meta_split_path"], manifest["meta_split_path"])
            self.assertEqual(grouping_payload["search_score_source"], "final_predictions")
            self.assertEqual(
                grouping_payload["search_score_path"],
                os.path.join(tmpdir, "final_predictions.json"),
            )
            self.assertEqual(partition_payload["search_score_source"], "final_predictions")
            self.assertEqual(
                partition_payload["search_score_path"],
                os.path.join(tmpdir, "final_predictions.json"),
            )

    def test_run_stage1_pipeline_group_proxy_search_skips_predictor_chain_and_writes_suffixed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            config.defrost()
            config.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE = "group_proxy"
            config.freeze()

            logger = mock.Mock()
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "selection_eval_label": "meta-val",
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }
            directed_affinity = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)

            with mock.patch(
                "ag_mtlora.stage1.build_stage1_data_split_manifest",
                return_value=manifest,
            ), mock.patch(
                "ag_mtlora.stage1.warmup_and_collect_affinity",
                return_value={
                    "directed_affinity": directed_affinity,
                    "symmetric_affinity": 0.5 * (directed_affinity + directed_affinity.T),
                    "num_affinity_epochs": 1,
                    "num_batches_per_epoch": [3],
                    "affinity_epoch_history": [directed_affinity],
                    "warmup_checkpoint_path": os.path.join(tmpdir, "warmup_checkpoint.pth"),
                    "post_affinity_checkpoint_path": os.path.join(tmpdir, "post_affinity_checkpoint.pth"),
                    "warmup_validation_losses": {"task_a": 1.0, "task_b": 2.0},
                    "post_affinity_validation_losses": {"task_a": 0.8, "task_b": 1.7},
                    "post_affinity_state_dict": {},
                },
            ), mock.patch(
                "ag_mtlora.stage1.create_resolved_training_config",
                return_value=DummyResolvedConfig(),
            ), mock.patch(
                "ag_mtlora.stage1.collect_predictor_training_targets"
            ) as mocked_collect, mock.patch(
                "ag_mtlora.stage1.fit_base_predictor"
            ) as mocked_fit_base, mock.patch(
                "ag_mtlora.stage1.fit_residual_predictor"
            ) as mocked_fit_residual, mock.patch(
                "ag_mtlora.stage1.predict_all_candidate_groups"
            ) as mocked_predict_all, mock.patch(
                "ag_mtlora.stage1.save_matrix_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_affinity_epoch_history_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_group_task_rows"
            ):
                artifacts = stage1.run_stage1_pipeline(
                    config,
                    tmpdir,
                    logger,
                    base_cfg_path=os.path.join(tmpdir, "base_config.yaml"),
                )

            mocked_collect.assert_not_called()
            mocked_fit_base.assert_not_called()
            mocked_fit_residual.assert_not_called()
            mocked_predict_all.assert_not_called()
            self.assertEqual(artifacts["search_score_source"], "group_proxy")
            self.assertEqual(artifacts["search_score_path"], os.path.join(tmpdir, "group_proxy.json"))
            self.assertIsNone(artifacts["predictor_train_groups_json"])
            self.assertIsNone(artifacts["initial_predictions_json"])
            self.assertIsNone(artifacts["final_predictions_json"])
            self.assertTrue(artifacts["grouping_json_path"].endswith("grouping__group_proxy.json"))
            self.assertTrue(artifacts["partition_results_path"].endswith("partition_search_results__group_proxy.json"))

            with open(artifacts["grouping_json_path"], "r", encoding="utf-8") as handle:
                grouping_payload = json.load(handle)
            with open(artifacts["partition_results_path"], "r", encoding="utf-8") as handle:
                partition_payload = json.load(handle)

            self.assertEqual(grouping_payload["search_score_source"], "group_proxy")
            self.assertEqual(grouping_payload["search_score_path"], os.path.join(tmpdir, "group_proxy.json"))
            self.assertIsNone(grouping_payload["final_predictions_path"])
            self.assertEqual(partition_payload["search_score_source"], "group_proxy")
            self.assertEqual(partition_payload["search_score_path"], os.path.join(tmpdir, "group_proxy.json"))
            self.assertEqual(partition_payload["ranked_partitions"][0]["groups"], [["task_a", "task_b"]])

    def test_write_search_artifacts_stage_mode_records_stagewise_groups_and_zeroes_inactive_slots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                handle.write("MODEL:\n  NAME: stage_test\n")

            search_scores_by_stage = [
                {
                    "task_a": {"task_a": 0.0},
                    "task_b": {"task_b": 0.0},
                    "task_a|task_b": {"task_a": 1.0, "task_b": 1.0},
                },
                {
                    "task_a": {"task_a": 0.6},
                    "task_b": {"task_b": 0.7},
                    "task_a|task_b": {"task_a": -0.1, "task_b": -0.2},
                },
            ]

            artifacts = stage1.write_search_artifacts(
                tasks=["task_a", "task_b"],
                search_scores=search_scores_by_stage,
                search_score_source="group_proxy",
                search_score_path=os.path.join(tmpdir, "group_proxy.json"),
                output_root=tmpdir,
                grouping_save_path=os.path.join(tmpdir, "grouping.json"),
                max_groups=2,
                group_shared_ranks=[],
                total_shared_rank_budget=4,
                num_stages=2,
                group_rank_allocation="equal_split",
                search_objective="mean_group_proxy",
                selection_data_split_mode="train_meta_strict",
                selection_eval_label="meta-val",
                meta_split_path=os.path.join(tmpdir, "meta_split.json"),
                affinity_path=os.path.join(tmpdir, "affinity.json"),
                final_predictions_path=None,
                warmup_checkpoint_path=os.path.join(tmpdir, "warmup_checkpoint.pth"),
                post_affinity_checkpoint_path=os.path.join(tmpdir, "post_affinity_checkpoint.pth"),
                affinity_warmup_epochs=5,
                affinity_score_epochs=10,
                base_cfg_path=base_cfg_path,
                runtime_snapshot_text="MODEL:\n  AGMTLORA:\n    ENABLED: true\n",
                logger=mock.Mock(),
                partition_granularity="stage",
            )

            self.assertEqual(
                [item["groups"] for item in artifacts["best_partition_by_stage"]],
                [[["task_a", "task_b"]], [["task_a"], ["task_b"]]],
            )
            self.assertEqual(artifacts["resolved_group_ranks"], [[4, 2], [0, 2]])

            with open(artifacts["grouping_json_path"], "r", encoding="utf-8") as handle:
                grouping_payload = json.load(handle)
            with open(artifacts["partition_results_path"], "r", encoding="utf-8") as handle:
                partition_payload = json.load(handle)

            self.assertEqual(grouping_payload["partition_granularity"], "stage")
            self.assertEqual(grouping_payload["group_slot_names"], ["group_0", "group_1"])
            self.assertEqual(
                grouping_payload["groups_by_stage"],
                [[["task_a", "task_b"]], [["task_a"], ["task_b"]]],
            )
            self.assertEqual(
                grouping_payload["task_to_group_by_stage"],
                [
                    {"task_a": "group_0", "task_b": "group_0"},
                    {"task_a": "group_0", "task_b": "group_1"},
                ],
            )
            self.assertEqual(grouping_payload["num_groups_by_stage"], [1, 2])
            self.assertEqual(partition_payload["ranked_partitions_by_stage"][0][0]["groups"], [["task_a", "task_b"]])
            self.assertEqual(partition_payload["ranked_partitions_by_stage"][1][0]["groups"], [["task_a"], ["task_b"]])

    def test_run_stage1_pipeline_stage_group_proxy_search_writes_stagewise_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            config.defrost()
            config.MODEL.SWIN.DEPTHS = [2, 2]
            config.MODEL.AGMTLORA.PARTITION_GRANULARITY = "stage"
            config.MODEL.AGMTLORA.SEARCH_SCORE_SOURCE = "group_proxy"
            config.freeze()

            base_cfg_path = os.path.join(tmpdir, "base_config.yaml")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                handle.write("MODEL:\n  NAME: stage_pipeline_test\n")

            logger = mock.Mock()
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "selection_eval_label": "meta-val",
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }
            directed_affinity_by_stage = [
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32),
            ]
            directed_affinity = np.mean(np.stack(directed_affinity_by_stage, axis=0), axis=0)

            with mock.patch(
                "ag_mtlora.stage1.build_stage1_data_split_manifest",
                return_value=manifest,
            ), mock.patch(
                "ag_mtlora.stage1.warmup_and_collect_affinity",
                return_value={
                    "directed_affinity": directed_affinity,
                    "directed_affinity_by_stage": directed_affinity_by_stage,
                    "symmetric_affinity": 0.5 * (directed_affinity + directed_affinity.T),
                    "num_affinity_epochs": 1,
                    "num_batches_per_epoch": [3],
                    "affinity_epoch_history": [directed_affinity],
                    "warmup_checkpoint_path": os.path.join(tmpdir, "warmup_checkpoint.pth"),
                    "post_affinity_checkpoint_path": os.path.join(tmpdir, "post_affinity_checkpoint.pth"),
                    "warmup_validation_losses": {"task_a": 1.0, "task_b": 2.0},
                    "post_affinity_validation_losses": {"task_a": 0.8, "task_b": 1.7},
                    "post_affinity_state_dict": {},
                },
            ):
                artifacts = stage1.run_stage1_pipeline(
                    config,
                    tmpdir,
                    logger,
                    base_cfg_path=base_cfg_path,
                )

            with open(artifacts["group_proxy_json_path"], "r", encoding="utf-8") as handle:
                group_proxy_payload = json.load(handle)
            with open(artifacts["grouping_json_path"], "r", encoding="utf-8") as handle:
                grouping_payload = json.load(handle)
            with open(artifacts["partition_results_path"], "r", encoding="utf-8") as handle:
                partition_payload = json.load(handle)

            self.assertEqual(group_proxy_payload["partition_granularity"], "stage")
            self.assertEqual(len(group_proxy_payload["group_proxy_by_stage"]), 2)
            self.assertEqual(grouping_payload["partition_granularity"], "stage")
            self.assertEqual(grouping_payload["groups_by_stage"], [[["task_a", "task_b"]], [["task_a"], ["task_b"]]])
            self.assertEqual(grouping_payload["group_shared_ranks"], [[2, 1], [0, 1]])
            self.assertEqual(len(partition_payload["ranked_partitions_by_stage"]), 2)
            self.assertTrue(artifacts["grouping_json_path"].endswith("grouping__group_proxy.json"))
            self.assertTrue(artifacts["partition_results_path"].endswith("partition_search_results__group_proxy.json"))

    def test_replay_stage1_partition_search_stage_group_proxy_reuses_stagewise_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                handle.write("MODEL:\n  NAME: replay_stage_test\n")

            group_proxy_by_stage = [
                {
                    "task_a": {"task_a": 0.0},
                    "task_b": {"task_b": 0.0},
                    "task_a|task_b": {"task_a": 0.9, "task_b": 0.8},
                },
                {
                    "task_a": {"task_a": 0.4},
                    "task_b": {"task_b": 0.5},
                    "task_a|task_b": {"task_a": -0.2, "task_b": -0.1},
                },
            ]
            with open(os.path.join(tmpdir, "group_proxy.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": ["task_a", "task_b"],
                        "partition_granularity": "stage",
                        "selection_data_split_mode": "train_meta_strict",
                        "selection_eval_label": "meta-val",
                        "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
                        "group_proxy": stage1.average_search_scores_by_stage(group_proxy_by_stage),
                        "group_proxy_by_stage": group_proxy_by_stage,
                    },
                    handle,
                    indent=2,
                )

            with open(os.path.join(tmpdir, "grouping.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": ["task_a", "task_b"],
                        "partition_granularity": "global",
                        "groups": [["task_a"], ["task_b"]],
                        "task_to_group": {"task_a": "group_0", "task_b": "group_1"},
                        "max_groups": 2,
                        "group_shared_ranks": [[1, 1], [1, 1]],
                        "search_score_source": "final_predictions",
                        "search_score_path": os.path.join(tmpdir, "final_predictions.json"),
                    },
                    handle,
                    indent=2,
                )
            with open(os.path.join(tmpdir, "partition_search_results.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": ["task_a", "task_b"],
                        "partition_granularity": "global",
                        "search_score_source": "final_predictions",
                        "ranked_partitions": [
                            {
                                "groups": [["task_a"], ["task_b"]],
                                "partition_score": 0.0,
                                "per_task_scores": {"task_a": 0.0, "task_b": 0.0},
                                "num_groups": 2,
                            }
                        ],
                    },
                    handle,
                    indent=2,
                )
            with open(os.path.join(tmpdir, "resolved_agmtlora_runtime_snapshot.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "MODEL": {
                            "SWIN": {"DEPTHS": [2, 2]},
                            "AGMTLORA": {
                                "MAX_GROUPS": 2,
                                "TOTAL_SHARED_RANK_BUDGET": 2,
                                "GROUP_SHARED_RANKS": [],
                                "GROUP_RANK_ALLOCATION": "equal_split",
                                "SEARCH_OBJECTIVE": "mean_final_predicted_gain",
                                "SEARCH_SCORE_SOURCE": "final_predictions",
                                "PARTITION_GRANULARITY": "global",
                            },
                        }
                    },
                    handle,
                    sort_keys=False,
                )

            stage1.write_search_artifacts(
                tasks=["task_a", "task_b"],
                search_scores=group_proxy_by_stage,
                search_score_source="group_proxy",
                search_score_path=os.path.join(tmpdir, "group_proxy.json"),
                output_root=tmpdir,
                grouping_save_path=os.path.join(tmpdir, "grouping__group_proxy.json"),
                max_groups=2,
                group_shared_ranks=[],
                total_shared_rank_budget=2,
                num_stages=2,
                group_rank_allocation="equal_split",
                search_objective="mean_group_proxy",
                selection_data_split_mode="train_meta_strict",
                selection_eval_label="meta-val",
                meta_split_path=os.path.join(tmpdir, "meta_split.json"),
                affinity_path=os.path.join(tmpdir, "affinity.json"),
                final_predictions_path=None,
                warmup_checkpoint_path=os.path.join(tmpdir, "warmup_checkpoint.pth"),
                post_affinity_checkpoint_path=os.path.join(tmpdir, "post_affinity_checkpoint.pth"),
                affinity_warmup_epochs=5,
                affinity_score_epochs=10,
                base_cfg_path=base_cfg_path,
                runtime_snapshot_text=yaml.safe_dump(
                    {
                        "MODEL": {
                            "SWIN": {"DEPTHS": [2, 2]},
                            "AGMTLORA": {
                                "MAX_GROUPS": 2,
                                "TOTAL_SHARED_RANK_BUDGET": 2,
                                "GROUP_SHARED_RANKS": [],
                                "GROUP_RANK_ALLOCATION": "equal_split",
                                "SEARCH_OBJECTIVE": "mean_group_proxy",
                                "SEARCH_SCORE_SOURCE": "group_proxy",
                                "PARTITION_GRANULARITY": "stage",
                            },
                        }
                    },
                    sort_keys=False,
                ),
                logger=mock.Mock(),
                partition_granularity="stage",
            )

            artifacts = stage1.replay_stage1_partition_search(
                base_cfg_path=base_cfg_path,
                stage1_dir=tmpdir,
                search_score_source="group_proxy",
                logger=mock.Mock(),
            )

            self.assertEqual(artifacts["partition_granularity"], "stage")
            self.assertEqual(
                [item["groups"] for item in artifacts["best_partition_by_stage"]],
                [[["task_a", "task_b"]], [["task_a"], ["task_b"]]],
            )
            self.assertEqual(artifacts["grouping_json_path"], os.path.join(tmpdir, "grouping__group_proxy.json"))
            self.assertEqual(artifacts["partition_results_path"], os.path.join(tmpdir, "partition_search_results__group_proxy.json"))

    def test_replay_stage1_partition_search_stage_mode_rejects_final_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                handle.write("MODEL:\n  NAME: replay_stage_error_test\n")

            with open(os.path.join(tmpdir, "grouping.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": ["task_a", "task_b"],
                        "partition_granularity": "stage",
                        "group_slot_names": ["group_0", "group_1"],
                        "groups_by_stage": [[["task_a", "task_b"]], [["task_a"], ["task_b"]]],
                    },
                    handle,
                    indent=2,
                )
            with open(os.path.join(tmpdir, "resolved_agmtlora_runtime_snapshot.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {"MODEL": {"SWIN": {"DEPTHS": [2, 2]}, "AGMTLORA": {"PARTITION_GRANULARITY": "stage"}}},
                    handle,
                    sort_keys=False,
                )

            with self.assertRaisesRegex(ValueError, "Stage-wise replay search only supports"):
                stage1.replay_stage1_partition_search(
                    base_cfg_path=base_cfg_path,
                    stage1_dir=tmpdir,
                    search_score_source="final_predictions",
                    logger=mock.Mock(),
                )

    def test_replay_stage1_partition_search_writes_side_by_side_proxy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = ["task_a", "task_b"]
            base_cfg_path = os.path.join(tmpdir, "base.yaml")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                handle.write("MODEL:\n  NAME: replay_test\n")

            group_proxy = {
                "task_a": {"task_a": 0.0},
                "task_b": {"task_b": 0.0},
                "task_a|task_b": {"task_a": 0.9, "task_b": 0.8},
            }
            final_predictions = {
                "task_a": {"task_a": 0.0},
                "task_b": {"task_b": 0.0},
                "task_a|task_b": {"task_a": -0.1, "task_b": -0.2},
            }

            with open(os.path.join(tmpdir, "group_proxy.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": tasks,
                        "selection_data_split_mode": "train_meta_strict",
                        "selection_eval_label": "meta-val",
                        "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
                        "group_proxy": group_proxy,
                    },
                    handle,
                    indent=2,
                )
            with open(os.path.join(tmpdir, "final_predictions.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": tasks,
                        "selection_data_split_mode": "train_meta_strict",
                        "selection_eval_label": "meta-val",
                        "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
                        "final_predictions": final_predictions,
                    },
                    handle,
                    indent=2,
                )

            original_ranked_partitions = stage1.run_partition_search(tasks, final_predictions, max_groups=2)
            original_grouping_payload = {
                "stage1_runtime_schema_version": stage1.STAGE1_RUNTIME_SCHEMA_VERSION,
                "tasks": tasks,
                "groups": [["task_a"], ["task_b"]],
                "task_to_group": {"task_a": "group_0", "task_b": "group_1"},
                "max_groups": 2,
                "partition_score": 0.0,
                "group_shared_ranks": [[1], [1]],
                "search_objective": "mean_final_predicted_gain",
                "search_score_source": "final_predictions",
                "search_score_path": os.path.join(tmpdir, "final_predictions.json"),
                "selection_data_split_mode": "train_meta_strict",
                "selection_eval_label": "meta-val",
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
                "affinity_path": os.path.join(tmpdir, "affinity.json"),
                "final_predictions_path": os.path.join(tmpdir, "final_predictions.json"),
                "warmup_checkpoint": os.path.join(tmpdir, "warmup_checkpoint.pth"),
                "post_affinity_checkpoint": os.path.join(tmpdir, "post_affinity_checkpoint.pth"),
                "affinity_warmup_epochs": 5,
                "affinity_score_epochs": 50,
                "partition_search_results": os.path.join(tmpdir, "partition_search_results.json"),
                "group_rank_source": "auto_equal_split",
            }
            with open(os.path.join(tmpdir, "grouping.json"), "w", encoding="utf-8") as handle:
                json.dump(original_grouping_payload, handle, indent=2)
            with open(os.path.join(tmpdir, "partition_search_results.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "tasks": tasks,
                        "selection_data_split_mode": "train_meta_strict",
                        "selection_eval_label": "meta-val",
                        "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
                        "search_objective": "mean_final_predicted_gain",
                        "search_score_source": "final_predictions",
                        "search_score_path": os.path.join(tmpdir, "final_predictions.json"),
                        "ranked_partitions": original_ranked_partitions,
                    },
                    handle,
                    indent=2,
                )
            with open(os.path.join(tmpdir, "resolved_agmtlora_runtime_snapshot.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "MODEL": {
                            "SWIN": {"DEPTHS": [2]},
                            "AGMTLORA": {
                                "MAX_GROUPS": 2,
                                "GROUP_SHARED_RANKS": [],
                                "TOTAL_SHARED_RANK_BUDGET": 2,
                                "GROUP_RANK_ALLOCATION": "equal_split",
                                "SEARCH_OBJECTIVE": "mean_final_predicted_gain",
                                "SEARCH_SCORE_SOURCE": "final_predictions",
                            },
                        }
                    },
                    handle,
                    sort_keys=False,
                )

            with open(os.path.join(tmpdir, "grouping.json"), "r", encoding="utf-8") as handle:
                original_grouping_text = handle.read()
            with open(os.path.join(tmpdir, "partition_search_results.json"), "r", encoding="utf-8") as handle:
                original_partition_text = handle.read()

            artifacts = stage1.replay_stage1_partition_search(
                base_cfg_path=base_cfg_path,
                stage1_dir=tmpdir,
                search_score_source="group_proxy",
                logger=mock.Mock(),
            )

            with open(os.path.join(tmpdir, "grouping.json"), "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), original_grouping_text)
            with open(os.path.join(tmpdir, "partition_search_results.json"), "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), original_partition_text)

            self.assertTrue(artifacts["grouping_json_path"].endswith("grouping__group_proxy.json"))
            self.assertTrue(artifacts["partition_results_path"].endswith("partition_search_results__group_proxy.json"))
            self.assertEqual(artifacts["search_score_source"], "group_proxy")
            self.assertEqual(artifacts["best_partition"]["groups"], [["task_a", "task_b"]])

            with open(artifacts["grouping_json_path"], "r", encoding="utf-8") as handle:
                grouping_payload = json.load(handle)
            with open(artifacts["partition_results_path"], "r", encoding="utf-8") as handle:
                partition_payload = json.load(handle)

            self.assertEqual(grouping_payload["search_score_source"], "group_proxy")
            self.assertEqual(grouping_payload["search_score_path"], os.path.join(tmpdir, "group_proxy.json"))
            self.assertIsNone(grouping_payload["final_predictions_path"])
            self.assertEqual(partition_payload["ranked_partitions"][0]["groups"], [["task_a", "task_b"]])
            self.assertEqual(
                artifacts["ranking_changes"][0],
                {"partition_key": "task_a|task_b", "old_rank": 2, "new_rank": 1},
            )


if __name__ == "__main__":
    unittest.main()
