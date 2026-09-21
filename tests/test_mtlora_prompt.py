import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from yacs.config import CfgNode as CN

from config import get_config
from models.build import build_model
from models.lora import (
    MTLoRALinear,
    RankExtractionGate,
    mark_only_lora_as_trainable,
    mark_prompt_as_trainable,
)
from models.swin_transformer_vpt import (
    PromptedSwinTransformer,
    PromptedWindowAttention,
    _aggregate_prompt_windows,
    _expand_prompt_windows,
)
from utils import _rank_extract_checkpoint_mode


TASKS = ["semseg", "sal"]


def make_args(cfg_path, tasks="semseg,sal"):
    return SimpleNamespace(
        cfg=cfg_path,
        opts=None,
        batch_size=None,
        ckpt_freq=None,
        eval_freq=None,
        skip_initial_validation=False,
        eval_training_freq=None,
        epochs=None,
        mti=None,
        decoder_map=None,
        skip_decoder=False,
        data_path=None,
        nyud=None,
        pascal="PASCAL_MT",
        tasks=tasks,
        zip=False,
        cache_mode=None,
        pretrained=None,
        resume=None,
        resume_backbone=False,
        freeze_backbone=False,
        save_sample=False,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level=None,
        disable_amp=False,
        output="",
        tag=None,
        eval=False,
        throughput=False,
        seed=None,
        deterministic=False,
        debug_repro_steps=None,
        enable_amp=False,
        fused_window_process=False,
        fused_layernorm=False,
        optim=None,
        name=None,
        local_rank=0,
    )


def make_prompt_yaml(location="prepend", num_tokens=4, deep=True):
    return f"""
DATA:
  IMG_SIZE: 32
MODEL:
  TYPE: swin
  NAME: prompt_test
  MTLORA:
    ENABLED: True
    R: [2]
    SHARED_SCALE: [1.0]
    TASK_SCALE: [1.0]
    DROPOUT: [0.0]
    R_PER_TASK:
      semseg: [1]
      sal: [1]
      shared: [2]
  PROMPT:
    ENABLED: True
    NUM_TOKENS: {num_tokens}
    DEEP: {str(deep)}
    LOCATION: {location}
    DROPOUT: 0.0
    INITIATION: random
"""


def make_rank_extract_yaml(grouping_json, grouping_source="fixed_json", task_rank=0):
    return f"""
DATA:
  IMG_SIZE: 32
MODEL:
  TYPE: swin
  NAME: rank_extract_test
  MTLORA:
    ENABLED: True
    R: [4]
    SHARED_SCALE: [1.0]
    TASK_SCALE: [1.0]
    DROPOUT: [0.0]
    R_PER_TASK:
      semseg: [{task_rank}]
      sal: [{task_rank}]
      shared: [4]
    RANK_EXTRACT:
      ENABLED: True
      STAGES: [2, 3]
      MODULES: [fc1]
      BLOCKS: all
      HIDDEN_DIM: 16
  PROMPT:
    ENABLED: True
    NUM_TOKENS: 2
    DEEP: True
    LOCATION: prepend
    DROPOUT: 0.0
    INITIATION: random
  AGMTLORA:
    ENABLED: True
    GROUPING_SOURCE: {grouping_source}
    GROUPING_JSON: {json.dumps(grouping_json)}
    GROUP_SHARED_RANKS: [[2], [2]]
"""


def make_mtlora(tasks=TASKS, ag_enabled=False):
    depths = [1, 1, 1, 1]
    cfg = CN(new_allowed=True)
    cfg.ENABLED = True
    cfg.BIAS = "none"
    cfg.R = [2, 2, 2, 2]
    cfg.SHARED_SCALE = [1.0, 1.0, 1.0, 1.0]
    cfg.TASK_SCALE = [1.0, 1.0, 1.0, 1.0]
    cfg.DROPOUT = [0.0, 0.0, 0.0, 0.0]
    cfg.TRAINABLE_SCALE_SHARED = False
    cfg.TRAINABLE_SCALE_PER_TASK = False
    cfg.INTERMEDIATE_SPECIALIZATION = False
    cfg.FREEZE_PRETRAINED = True
    cfg.SPLIT_QKV = False
    cfg.SHARED_MODE = "matrix"
    cfg.QKV_ENABLED = True
    cfg.PROJ_ENABLED = True
    cfg.FC1_ENABLED = True
    cfg.FC2_ENABLED = True
    cfg.DOWNSAMPLER_ENABLED = False
    cfg.R_PER_TASK_LIST = []
    cfg.SCALE_PER_TASK_LIST = []
    for _ in depths:
        layer_r = {"shared": 2}
        layer_scale = {}
        for task in tasks:
            layer_r[task] = 1
            layer_scale[task] = 1.0
        cfg.R_PER_TASK_LIST.append(layer_r)
        cfg.SCALE_PER_TASK_LIST.append(layer_scale)

    cfg.AGMTLORA_ENABLED = bool(ag_enabled)
    cfg.AGMTLORA_STAGE = 1 if ag_enabled else 0
    cfg.AGMTLORA_PARTITION_GRANULARITY = "global"
    cfg.AGMTLORA_GROUPS = []
    cfg.AGMTLORA_GROUP_NAMES = []
    cfg.AGMTLORA_GROUP_RANKS = []
    cfg.AGMTLORA_TASK_TO_GROUP = CN(new_allowed=True)
    cfg.AGMTLORA_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)
    cfg.RANK_EXTRACT = CN()
    cfg.RANK_EXTRACT.ENABLED = False
    cfg.RANK_EXTRACT.STAGES = [2, 3]
    cfg.RANK_EXTRACT.MODULES = ["fc1"]
    cfg.RANK_EXTRACT.BLOCKS = "all"
    cfg.RANK_EXTRACT.HIDDEN_DIM = 16
    if ag_enabled:
        cfg.AGMTLORA_GROUP_NAMES = ["group_0", "group_1"]
        cfg.AGMTLORA_GROUP_RANKS = [[2, 2, 2, 2], [2, 2, 2, 2]]
        cfg.AGMTLORA_TASK_TO_GROUP["semseg"] = "group_0"
        cfg.AGMTLORA_TASK_TO_GROUP["sal"] = "group_1"
    return cfg


def make_model_config(ag_enabled=False):
    cfg = CN(new_allowed=True)
    cfg.FUSED_LAYERNORM = False
    cfg.FUSED_WINDOW_PROCESS = False
    cfg.TASKS = list(TASKS)
    cfg.DATA = CN(new_allowed=True)
    cfg.DATA.IMG_SIZE = [32, 32]
    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.TYPE = "swin"
    cfg.MODEL.NUM_CLASSES = 0
    cfg.MODEL.DROP_RATE = 0.0
    cfg.MODEL.DROP_PATH_RATE = 0.0
    cfg.MODEL.SWIN = CN(new_allowed=True)
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.IN_CHANS = 3
    cfg.MODEL.SWIN.EMBED_DIM = 8
    cfg.MODEL.SWIN.DEPTHS = [1, 1, 1, 1]
    cfg.MODEL.SWIN.NUM_HEADS = [1, 2, 4, 8]
    cfg.MODEL.SWIN.WINDOW_SIZE = 2
    cfg.MODEL.SWIN.MLP_RATIO = 2.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.MTLORA = make_mtlora(ag_enabled=ag_enabled)
    cfg.MODEL.PROMPT = CN(new_allowed=True)
    cfg.MODEL.PROMPT.ENABLED = True
    cfg.MODEL.PROMPT.NUM_TOKENS = 2
    cfg.MODEL.PROMPT.DEEP = True
    cfg.MODEL.PROMPT.LOCATION = "prepend"
    cfg.MODEL.PROMPT.DROPOUT = 0.0
    cfg.MODEL.PROMPT.INITIATION = "random"
    cfg.TRAIN = CN(new_allowed=True)
    cfg.TRAIN.USE_CHECKPOINT = False
    return cfg


class MTLoRAPromptTest(unittest.TestCase):
    def test_prompt_window_round_trip_preserves_batch_ownership(self):
        prompt_emb = torch.tensor([[[0.0]], [[1.0]]])

        prompt_windows = _expand_prompt_windows(prompt_emb, num_windows=3)

        torch.testing.assert_close(
            prompt_windows[:, 0, 0],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        )
        aggregated = _aggregate_prompt_windows(
            prompt_windows, batch_size=2, num_windows=3
        )
        torch.testing.assert_close(aggregated, prompt_emb)

    def test_prompt_window_aggregation_keeps_task_branch_samples_independent(self):
        task_prompt_windows = torch.tensor(
            [[[10.0]], [[11.0]], [[12.0]], [[20.0]], [[21.0]], [[22.0]]]
        )

        aggregated = _aggregate_prompt_windows(
            task_prompt_windows, batch_size=2, num_windows=3
        )

        torch.testing.assert_close(
            aggregated[:, 0, 0], torch.tensor([11.0, 21.0])
        )

    def test_config_accepts_minimal_prompt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = os.path.join(tmp_dir, "prompt.yaml")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                handle.write(make_prompt_yaml())

            cfg = get_config(make_args(cfg_path))

        self.assertTrue(cfg.MODEL.PROMPT.ENABLED)
        self.assertEqual(cfg.MODEL.PROMPT.LOCATION, "prepend")
        self.assertEqual(cfg.MODEL.PROMPT.NUM_TOKENS, 4)

    def test_config_rejects_unsupported_prompt_location(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = os.path.join(tmp_dir, "prompt.yaml")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                handle.write(make_prompt_yaml(location="add"))

            with self.assertRaisesRegex(ValueError, "LOCATION"):
                get_config(make_args(cfg_path))

    def test_config_accepts_rank_extraction_with_fixed_groups_and_zero_task_ranks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            grouping_path = os.path.join(tmp_dir, "grouping.json")
            with open(grouping_path, "w", encoding="utf-8") as handle:
                json.dump({"tasks": TASKS, "groups": [["semseg"], ["sal"]]}, handle)
            cfg_path = os.path.join(tmp_dir, "rank_extract.yaml")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                handle.write(make_rank_extract_yaml(grouping_path))

            cfg = get_config(make_args(cfg_path))

        self.assertTrue(cfg.MODEL.MTLORA.RANK_EXTRACT.ENABLED)
        self.assertEqual(cfg.MODEL.MTLORA.RANK_EXTRACT.STAGES, [2, 3])
        self.assertTrue(cfg.MODEL.MTLORA.AGMTLORA_ENABLED)

    def test_config_rejects_rank_extraction_during_group_search(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = os.path.join(tmp_dir, "rank_extract.yaml")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                handle.write(make_rank_extract_yaml("", grouping_source="search"))

            with self.assertRaisesRegex(ValueError, "fixed_json"):
                get_config(make_args(cfg_path))

    def test_config_rejects_rank_extraction_with_task_lora(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            grouping_path = os.path.join(tmp_dir, "grouping.json")
            with open(grouping_path, "w", encoding="utf-8") as handle:
                json.dump({"tasks": TASKS, "groups": [["semseg"], ["sal"]]}, handle)
            cfg_path = os.path.join(tmp_dir, "rank_extract.yaml")
            with open(cfg_path, "w", encoding="utf-8") as handle:
                handle.write(make_rank_extract_yaml(grouping_path, task_rank=1))

            with self.assertRaisesRegex(ValueError, "task-specific LoRA rank"):
                get_config(make_args(cfg_path))

    def test_build_model_uses_prompted_swin_when_enabled(self):
        model = build_model(make_model_config())

        self.assertIsInstance(model, PromptedSwinTransformer)

    def test_prompted_backbone_forward_returns_cropped_stage_features(self):
        model = build_model(make_model_config(ag_enabled=True))
        model.eval()
        x = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            features = model(x, task="semseg", return_stages=True)

        self.assertEqual(len(features), 4)
        self.assertEqual([feature.shape[1] for feature in features], [16, 4, 1, 1])

    def test_prompted_backbone_eval_is_batch_and_reorder_independent(self):
        model = build_model(make_model_config(ag_enabled=True))
        model.eval()
        images = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            batched = model(images, task="semseg", return_stages=True)
            reordered = model(images.flip(0), task="semseg", return_stages=True)
            singles = [
                model(images[index:index + 1], task="semseg", return_stages=True)
                for index in range(2)
            ]

        for stage_idx, batched_stage in enumerate(batched):
            torch.testing.assert_close(batched_stage[0:1], singles[0][stage_idx])
            torch.testing.assert_close(batched_stage[1:2], singles[1][stage_idx])
            torch.testing.assert_close(batched_stage, reordered[stage_idx].flip(0))

    def test_prompted_backbone_builds_rank_extractors_only_for_target_fc1_stages(self):
        config = make_model_config(ag_enabled=True)
        config.MODEL.MTLORA.defrost()
        for stage_ranks in config.MODEL.MTLORA.R_PER_TASK_LIST:
            for task in TASKS:
                stage_ranks[task] = 0
        config.MODEL.MTLORA.RANK_EXTRACT.ENABLED = True
        config.MODEL.MTLORA.RANK_EXTRACT.STAGES = [2, 3]
        config.MODEL.MTLORA.RANK_EXTRACT.HIDDEN_DIM = 16
        config.MODEL.MTLORA.freeze()

        model = build_model(config)
        target_linears = [
            module for module in model.modules()
            if isinstance(module, MTLoRALinear) and module.rank_extract_enabled
        ]
        gates = [module for module in model.modules() if isinstance(module, RankExtractionGate)]

        self.assertEqual(len(target_linears), 2)
        self.assertTrue(all(set(module.lora_rank_extractors) == {"group_0", "group_1"}
                            for module in target_linears))
        self.assertEqual(len(gates), 4)
        self.assertEqual(sum(param.numel() for gate in gates for param in gate.parameters()), 456)

    def test_prompt_attention_keeps_ag_task_qkv_branches(self):
        attention = PromptedWindowAttention(
            num_prompts=2,
            dim=8,
            window_size=(2, 2),
            num_heads=2,
            tasks=TASKS,
            mtlora=make_mtlora(ag_enabled=True),
            layer_idx=0,
            lora=True,
        )
        calls = []
        original_apply_attention = attention._apply_attention_from_qkv

        def wrapped_apply_attention(qkv, batch_windows, token_count, channels, mask=None):
            calls.append(qkv)
            return original_apply_attention(qkv, batch_windows, token_count, channels, mask=mask)

        attention._apply_attention_from_qkv = wrapped_apply_attention
        x = torch.randn(1, 6, 8)

        _, task_outputs = attention(x)

        self.assertIsNotNone(task_outputs)
        self.assertEqual(set(task_outputs.keys()), set(TASKS))
        self.assertEqual(len(calls), 1 + len(TASKS))

    def test_ag_linear_active_task_matches_full_routing_branch(self):
        layer = MTLoRALinear(
            4,
            6,
            r={"group_0": 2, "group_1": 2},
            lora_shared_scale=1.0,
            lora_dropout=0.0,
            tasks=TASKS,
            task_to_group={"semseg": "group_0", "sal": "group_1"},
        )
        with torch.no_grad():
            for group in layer.lora_shared_B_groups.values():
                group.normal_()
        x = torch.randn(2, 3, 4)

        shared_output, all_task_outputs = layer(x)
        selected_shared_output, selected_task_outputs = layer(x, active_task="sal")

        torch.testing.assert_close(selected_shared_output, shared_output)
        self.assertEqual(set(selected_task_outputs), {"sal"})
        torch.testing.assert_close(selected_task_outputs["sal"], all_task_outputs["sal"])
        with self.assertRaisesRegex(ValueError, "active_task"):
            layer(x, active_task="unknown")

    def test_rank_extraction_gate_initializes_to_exact_identity(self):
        gate = RankExtractionGate(rank=3, hidden_dim=5)
        rank_activations = torch.randn(2, 4, 3)
        condition = torch.randn(2, 3)

        mask = gate(rank_activations, condition)

        torch.testing.assert_close(mask, torch.ones_like(mask), rtol=0.0, atol=0.0)

    def test_rank_extraction_identity_matches_group_lora_baseline(self):
        layer = MTLoRALinear(
            4,
            6,
            r={"group_0": 2, "group_1": 3},
            lora_shared_scale=1.25,
            lora_dropout=0.0,
            tasks=TASKS,
            task_to_group={"semseg": "group_0", "sal": "group_1"},
            rank_extract_enabled=True,
            rank_extract_hidden_dim=4,
        )
        with torch.no_grad():
            for group in layer.lora_shared_B_groups.values():
                group.normal_()
        x = torch.randn(2, 5, 4)
        x_tasks = {"semseg": torch.randn(2, 5, 4)}

        layer.rank_extract_enabled = False
        _, baseline = layer(x, x_tasks, active_task="semseg", prompt_token_count=2)
        layer.rank_extract_enabled = True
        _, extracted = layer(x, x_tasks, active_task="semseg", prompt_token_count=2)

        torch.testing.assert_close(extracted["semseg"], baseline["semseg"])

    def test_rank_extraction_identity_preserves_training_dropout_sequence(self):
        layer = MTLoRALinear(
            4,
            6,
            r={"group_0": 2},
            lora_shared_scale=1.0,
            lora_dropout=0.4,
            tasks=["semseg"],
            task_to_group={"semseg": "group_0"},
            rank_extract_enabled=True,
        )
        layer.train()
        with torch.no_grad():
            layer.lora_shared_B_groups["group_0"].normal_()
        x = torch.randn(2, 5, 4)
        task_input = torch.randn(2, 5, 4)

        layer.rank_extract_enabled = False
        torch.manual_seed(17)
        _, baseline = layer(x, {"semseg": task_input}, active_task="semseg", prompt_token_count=2)
        layer.rank_extract_enabled = True
        torch.manual_seed(17)
        _, extracted = layer(x, {"semseg": task_input}, active_task="semseg", prompt_token_count=2)

        torch.testing.assert_close(extracted["semseg"], baseline["semseg"])

    def test_rank_extraction_changes_only_patch_group_residual(self):
        layer = MTLoRALinear(
            3,
            4,
            r={"group_0": 2},
            lora_shared_scale=1.0,
            lora_dropout=0.0,
            tasks=["semseg"],
            task_to_group={"semseg": "group_0"},
            rank_extract_enabled=True,
            rank_extract_hidden_dim=3,
        )
        with torch.no_grad():
            layer.lora_shared_B_groups["group_0"].normal_()
        x = torch.randn(1, 5, 3)
        task_input = torch.randn(1, 5, 3)

        _, identity_output = layer(
            x,
            {"semseg": task_input},
            active_task="semseg",
            prompt_token_count=2,
        )
        with torch.no_grad():
            layer.lora_rank_extractors["group_0"].output.bias.fill_(1.0)
        _, modulated_output = layer(
            x,
            {"semseg": task_input},
            active_task="semseg",
            prompt_token_count=2,
        )

        torch.testing.assert_close(
            modulated_output["semseg"][:, :2], identity_output["semseg"][:, :2]
        )
        self.assertFalse(torch.allclose(
            modulated_output["semseg"][:, 2:], identity_output["semseg"][:, 2:]
        ))

    def test_rank_extraction_gate_responds_to_condition_and_rank_activation(self):
        gate = RankExtractionGate(rank=2, hidden_dim=2)
        with torch.no_grad():
            gate.z_proj.weight.copy_(torch.eye(2))
            gate.condition_proj.weight.copy_(torch.eye(2))
            gate.output.weight.copy_(torch.eye(2))
        z = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        condition = torch.tensor([[1.0, -1.0]])

        base = gate(z, condition)
        changed_condition = gate(z, torch.tensor([[-1.0, 1.0]]))
        changed_z = gate(z.flip(1), condition)

        self.assertFalse(torch.allclose(base, changed_condition))
        self.assertFalse(torch.allclose(base, changed_z))

    def test_rank_extraction_gradients_follow_only_selected_group(self):
        layer = MTLoRALinear(
            3,
            4,
            r={"group_0": 2, "group_1": 3},
            lora_shared_scale=1.0,
            lora_dropout=0.0,
            tasks=TASKS,
            task_to_group={"semseg": "group_0", "sal": "group_1"},
            rank_extract_enabled=True,
            rank_extract_hidden_dim=3,
        )
        with torch.no_grad():
            layer.lora_shared_B_groups["group_0"].normal_()
            layer.lora_rank_extractors["group_0"].output.weight.normal_()
        shared_input = torch.randn(1, 5, 3)
        task_input = torch.randn(1, 5, 3, requires_grad=True)

        _, outputs = layer(
            shared_input,
            {"semseg": task_input},
            active_task="semseg",
            prompt_token_count=2,
        )
        outputs["semseg"][:, 2:].sum().backward()

        self.assertGreater(task_input.grad[:, :2].abs().sum().item(), 0.0)
        self.assertIsNotNone(layer.lora_rank_extractors["group_0"].output.weight.grad)
        self.assertIsNone(layer.lora_rank_extractors["group_1"].output.weight.grad)
        self.assertIsNone(layer.lora_shared_A_groups["group_1"].grad)
        extractor_parameter_ids = {
            id(parameter)
            for parameter in layer.lora_rank_extractors.parameters()
        }
        self.assertNotIn(id(layer.lora_shared_A_groups["group_0"]), extractor_parameter_ids)

    def test_rank_extraction_rejects_missing_prompt_or_task_route(self):
        layer = MTLoRALinear(
            3,
            4,
            r={"group_0": 2},
            tasks=["semseg"],
            task_to_group={"semseg": "group_0"},
            rank_extract_enabled=True,
        )
        x = torch.randn(1, 4, 3)
        with self.assertRaisesRegex(ValueError, "active_task"):
            layer(x, prompt_token_count=1)
        with self.assertRaisesRegex(ValueError, "prompt_token_count"):
            layer(x, active_task="semseg", prompt_token_count=0)

    def test_rank_extraction_checkpoint_modes_distinguish_init_and_resume(self):
        target_state = {
            "backbone.fc1.lora_rank_extractors.group_0.output.weight": object(),
            "backbone.fc1.lora_rank_extractors.group_0.output.bias": object(),
        }

        mode, missing = _rank_extract_checkpoint_mode({}, target_state)
        self.assertEqual(mode, "initialization")
        self.assertEqual(missing, set(target_state))

        mode, missing = _rank_extract_checkpoint_mode(dict(target_state), target_state)
        self.assertEqual(mode, "resume")
        self.assertEqual(missing, set(target_state))

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _rank_extract_checkpoint_mode(
                {next(iter(target_state)): object()}, target_state
            )

    def test_prompt_attention_active_task_computes_only_selected_branch(self):
        attention = PromptedWindowAttention(
            num_prompts=2,
            dim=8,
            window_size=(2, 2),
            num_heads=2,
            tasks=TASKS,
            mtlora=make_mtlora(ag_enabled=True),
            layer_idx=0,
            lora=True,
        )
        calls = []
        original_apply_attention = attention._apply_attention_from_qkv

        def wrapped_apply_attention(qkv, batch_windows, token_count, channels, mask=None):
            calls.append(qkv)
            return original_apply_attention(qkv, batch_windows, token_count, channels, mask=mask)

        attention._apply_attention_from_qkv = wrapped_apply_attention
        x = torch.randn(1, 6, 8)

        _, all_task_outputs = attention(x)
        calls.clear()
        _, selected_task_outputs = attention(x, active_task="sal")

        self.assertEqual(set(selected_task_outputs), {"sal"})
        torch.testing.assert_close(selected_task_outputs["sal"], all_task_outputs["sal"])
        self.assertEqual(len(calls), 2)

    def test_prompted_backbone_propagates_active_task_to_attention(self):
        model = build_model(make_model_config(ag_enabled=True))
        model.eval()
        active_tasks = []
        for module in model.modules():
            if not isinstance(module, PromptedWindowAttention):
                continue
            original_forward = module.forward

            def wrapped_forward(*args, _forward=original_forward, **kwargs):
                active_tasks.append(kwargs.get("active_task"))
                return _forward(*args, **kwargs)

            module.forward = wrapped_forward

        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32), task="semseg", return_stages=True)

        self.assertGreater(len(active_tasks), 0)
        self.assertEqual(set(active_tasks), {"semseg"})

    def test_prompt_parameters_are_trainable_after_lora_freeze(self):
        model = build_model(make_model_config())
        prompt_params = [
            param for name, param in model.named_parameters()
            if "prompt_embeddings" in name or "deep_prompt_embeddings" in name
        ]
        self.assertGreater(len(prompt_params), 0)

        mark_only_lora_as_trainable(
            model,
            bias="none",
            freeze_patch_embed=True,
            freeze_norm=True,
            free_relative_bias=True,
            freeze_downsample_reduction=True,
        )
        self.assertTrue(all(not param.requires_grad for param in prompt_params))

        mark_prompt_as_trainable(model)

        self.assertTrue(all(param.requires_grad for param in prompt_params))


if __name__ == "__main__":
    unittest.main()
