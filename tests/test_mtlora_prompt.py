import os
import tempfile
import unittest
import copy
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import torch
from yacs.config import CfgNode as CN

from config import get_config, _C
from models.build import build_model
from models.lora import MTLoRALinear, mark_only_lora_as_trainable, mark_prompt_as_trainable
from models.swin_transformer_vpt import PromptedSwinTransformer, PromptedWindowAttention
from stage_ablation import (
    prepare_trainable_parameters, disabled_adapter_parameters, resolve_stage_modes,
    adapter_parameter_summary, stage_ablation_metadata, validate_ablation_checkpoint,
)
from optimizer import build_optimizer


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


VARIANTS = {"baseline": ["B"] * 4}
for _mode in ("P", "G"):
    for _stage in range(4):
        _modes = ["B"] * 4
        _modes[_stage] = _mode
        VARIANTS[f"S{_stage + 1}-{_mode}"] = _modes


def make_stage_config(modes, split_qkv=False):
    cfg = _C.clone()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(make_model_config(ag_enabled=True))
    cfg.DATA.IMG_SIZE = [64, 64]
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.MTLORA.SPLIT_QKV = split_qkv
    cfg.MODEL.MTLORA.TRAINABLE_SCALE_SHARED = True
    cfg.MODEL.MTLORA.R_PER_TASK_LIST = [{"shared": 2, "semseg": 0, "sal": 0} for _ in range(4)]
    cfg.MODEL.STAGE_ABLATION.ENABLED = True
    cfg.MODEL.STAGE_ABLATION.MODES = list(modes)
    cfg.MODEL.AGMTLORA.ENABLED = True
    cfg.MODEL.AGMTLORA.GROUPING_SOURCE = "fixed_json"
    cfg.MODEL.AGMTLORA.RESOLVED_GROUP_NAMES = ["group_0", "group_1"]
    cfg.MODEL.AGMTLORA.RESOLVED_GROUP_SHARED_RANKS = [[2] * 4, [2] * 4]
    cfg.MODEL.UPDATE_RELATIVE_POSITION = False
    cfg.MODEL.FREEZE_BACKBONE = False
    cfg.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP_BY_STAGE = CN(new_allowed=True)
    cfg.TRAIN.BASE_LR = 0.001
    cfg.MTL = False
    return cfg


def build_stage_model(cfg):
    with redirect_stdout(io.StringIO()):
        model = build_model(cfg)
        prepare_trainable_parameters(model, cfg)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MTLoRALinear) and hasattr(module, "lora_shared_B_groups"):
                for parameter in module.lora_shared_B_groups.values():
                    parameter.normal_(std=0.05)
    return model.eval()


class StageAblationTest(unittest.TestCase):
    def test_all_b_matches_legacy_with_same_initialization_and_keys(self):
        cfg = make_stage_config(VARIANTS["baseline"])
        torch.manual_seed(19)
        ablated = build_stage_model(cfg)
        cfg.MODEL.STAGE_ABLATION.ENABLED = False
        torch.manual_seed(19)
        legacy = build_stage_model(cfg)
        self.assertEqual(list(ablated.state_dict()), list(legacy.state_dict()))
        for key in legacy.state_dict():
            torch.testing.assert_close(ablated.state_dict()[key], legacy.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 3, 64, 64)
        for task in TASKS:
            for a, b in zip(ablated(x, task=task, return_stages=True), legacy(x, task=task, return_stages=True)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(ablated(x, task=task), legacy(x, task=task), rtol=0, atol=0)

    def test_nine_variants_forward_backward_tokens_routing_and_reload(self):
        from utils import load_checkpoint, save_checkpoint
        from experiment_utils import load_model_state

        class State:
            def __init__(self):
                self.value = 0
            def state_dict(self):
                return {"value": self.value}
            def load_state_dict(self, state):
                self.value = state["value"]

        for name, modes in VARIANTS.items():
            with self.subTest(variant=name):
                cfg = make_stage_config(modes)
                model = build_stage_model(cfg)
                optimizer = build_optimizer(cfg, model)
                disabled = list(disabled_adapter_parameters(model))
                excluded = {id(p) for p in disabled}
                self.assertTrue(all(not p.requires_grad for p in disabled))
                self.assertFalse(excluded & {id(p) for g in optimizer.param_groups for p in g["params"]})
                before = [p.detach().clone() for p in disabled]
                observations = []
                hooks = []
                group_calls = []
                for stage, layer in enumerate(model.layers):
                    for block in layer.blocks:
                        def observe(module, args, kwargs, s=stage):
                            observations.append((s, args[0].shape[1], kwargs["active_task"]))
                        hooks.append(block.attn.register_forward_pre_hook(observe, with_kwargs=True))
                    for module in layer.modules():
                        if isinstance(module, MTLoRALinear):
                            original = module._apply_group_shared_lora
                            def routed(group, tensor, s=stage, fn=original):
                                group_calls.append((s, group))
                                return fn(group, tensor)
                            module._apply_group_shared_lora = routed
                x = torch.randn(2, 3, 64, 64, requires_grad=True)
                for task in TASKS:
                    features = model(x, task=task, return_stages=True)
                    self.assertEqual([tuple(f.shape) for f in features], [(2, 64, 16), (2, 16, 32), (2, 4, 64), (2, 4, 64)])
                    sum(f.square().mean() for f in features).backward()
                for stage, tokens, task in observations:
                    expected = model.layers[stage].blocks[0].window_size ** 2 + (0 if modes[stage] == "G" else 2)
                    self.assertEqual(tokens, expected)
                    self.assertIn(task, TASKS)
                for stage, mode in enumerate(modes):
                    if mode == "P":
                        self.assertFalse(any(s == stage for s, _ in group_calls))
                    else:
                        self.assertEqual({g for s, g in group_calls if s == stage}, {"group_0", "group_1"})
                self.assertGreater(x.grad.abs().sum().item(), 0)
                self.assertGreater(model.patch_embed.proj.weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in disabled))
                for stage, mode in enumerate(modes):
                    if mode != "G":
                        self.assertGreater(model.deep_prompt_embeddings[stage]["semseg"].grad.abs().sum().item(), 0)
                    if mode != "P":
                        group_b = model.layers[stage].blocks[0].attn.qkv.lora_shared_B_groups["group_0"]
                        self.assertGreater(group_b.grad.abs().sum().item(), 0)
                optimizer.step()
                for old, parameter in zip(before, disabled):
                    torch.testing.assert_close(old, parameter, rtol=0, atol=0)
                for hook in hooks:
                    hook.remove()
                with torch.no_grad():
                    reference = model(x, task="semseg", return_stages=True)
                    for parameter in disabled:
                        parameter.add_(torch.randn_like(parameter) * 10)
                    perturbed = model(x, task="semseg", return_stages=True)
                for old, new in zip(reference, perturbed):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                mark_prompt_as_trainable(model)
                self.assertTrue(all(not p.requires_grad for p in disabled))
                scheduler, scaler = State(), State()
                scheduler.value, scaler.value = 12, 34
                with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                    cfg.OUTPUT = folder
                    path = save_checkpoint(cfg, 4, model, 0.0, optimizer, scheduler, scaler, logging.getLogger("test"))
                    cfg.MODEL.RESUME = path
                    restored = build_stage_model(cfg)
                    restored_optimizer = build_optimizer(cfg, restored)
                    restored_scheduler, restored_scaler = State(), State()
                    load_checkpoint(cfg, restored, restored_optimizer, restored_scheduler, restored_scaler, logging.getLogger("test"))
                    self.assertEqual(cfg.TRAIN.START_EPOCH, 5)
                    self.assertEqual(restored_scheduler.value, 12)
                    self.assertEqual(restored_scaler.value, 34)
                    self.assertTrue(restored_optimizer.state)
                    load_model_state(restored, path, cfg, logging.getLogger("test"))
                    for a, b in zip(reference, restored(x, task="semseg", return_stages=True)):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(adapter_parameter_summary(model), adapter_parameter_summary(restored))

    def test_split_qkv_disabled_scales_and_stage_boundaries(self):
        for modes in (["G", "G", "B", "G"], ["B", "P", "P", "B"], ["G"] * 4):
            cfg = make_stage_config(modes, split_qkv=True)
            model = build_stage_model(cfg)
            disabled = list(disabled_adapter_parameters(model))
            self.assertTrue(all(not p.requires_grad for p in disabled))
            result = model(torch.randn(2, 3, 64, 64), task="sal")
            self.assertEqual(tuple(result.shape), (2, 64))
            result.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in disabled))

    def test_invalid_policies_and_checkpoint_mismatch(self):
        for modes in (["B"], ["B", "X", "B", "B"]):
            with self.assertRaises(ValueError):
                resolve_stage_modes(make_stage_config(modes))
        cfg = make_stage_config(VARIANTS["baseline"])
        cfg.MODEL.MTLORA.R_PER_TASK_LIST[0]["semseg"] = 1
        with self.assertRaisesRegex(ValueError, "rank"):
            resolve_stage_modes(cfg)
        cfg = make_stage_config(VARIANTS["baseline"])
        cfg.MODEL.AGMTLORA.GROUPING_SOURCE = "search"
        with self.assertRaisesRegex(ValueError, "fixed_json"):
            resolve_stage_modes(cfg)
        cfg = make_stage_config(VARIANTS["baseline"])
        checkpoint = {"stage_ablation": stage_ablation_metadata(cfg)}
        validate_ablation_checkpoint(cfg, checkpoint)
        for change in ("mode", "grouping", "missing"):
            bad = copy.deepcopy(checkpoint)
            if change == "mode":
                bad["stage_ablation"]["modes"][0] = "P"
            elif change == "grouping":
                bad["stage_ablation"]["grouping_id"] = "wrong"
            else:
                bad = {}
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_ablation_checkpoint(cfg, bad)
        model = build_stage_model(cfg)
        optimizer = build_optimizer(cfg, model)
        with self.assertRaisesRegex(ValueError, "parameter groups"):
            validate_ablation_checkpoint(cfg, checkpoint, model, optimizer)
        from stage_ablation import optimizer_parameter_names
        checkpoint["optimizer_parameter_names"] = optimizer_parameter_names(model, optimizer)
        checkpoint.update(optimizer={}, lr_scheduler={}, epoch=0, scaler={})
        validate_ablation_checkpoint(cfg, checkpoint, model, optimizer)
        checkpoint["optimizer_parameter_names"][0].reverse()
        with self.assertRaisesRegex(ValueError, "parameter groups"):
            validate_ablation_checkpoint(cfg, checkpoint, model, optimizer)

    def test_resume_protocol_is_checked_but_eval_batch_size_may_change(self):
        from stage_ablation import optimizer_parameter_names
        from utils import scale_learning_rates
        cfg = make_stage_config(VARIANTS["baseline"])
        model = build_stage_model(cfg)
        optimizer = build_optimizer(cfg, model)
        checkpoint = {"stage_ablation": stage_ablation_metadata(cfg),
                      "optimizer_parameter_names": optimizer_parameter_names(model, optimizer),
                      "optimizer": {}, "lr_scheduler": {}, "epoch": 0, "scaler": {}}
        raw_metadata = stage_ablation_metadata(cfg)
        scale_learning_rates(cfg, 1)
        self.assertEqual(raw_metadata, stage_ablation_metadata(cfg))
        validate_ablation_checkpoint(cfg, checkpoint, model, optimizer)
        cfg.defrost()
        cfg.DATA.BATCH_SIZE = 1
        validate_ablation_checkpoint(cfg, checkpoint)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_ablation_checkpoint(cfg, checkpoint, model, optimizer)

    def test_common_global_weights_expand_to_fixed_group_ranks(self):
        from utils import load_checkpoint
        cfg = make_stage_config(["B", "B", "P", "B"])
        source_cfg = cfg.clone()
        source_cfg.MODEL.STAGE_ABLATION.ENABLED = False
        source_cfg.MODEL.AGMTLORA.ENABLED = False
        source_cfg.MODEL.MTLORA.AGMTLORA_ENABLED = False
        source = build_stage_model(source_cfg)
        target = build_stage_model(cfg)
        with torch.no_grad():
            for module in source.modules():
                if isinstance(module, MTLoRALinear) and hasattr(module, "lora_shared_B"):
                    module.lora_shared_B.fill_(0.25)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = os.path.join(folder, "global.pth")
            torch.save({"model": source.state_dict()}, path)
            load_checkpoint(cfg, target, None, None, None, logging.getLogger("test"), initialization_path=path)
            for name, module in target.named_modules():
                if isinstance(module, MTLoRALinear) and hasattr(module, "lora_shared_B_groups"):
                    for parameter in module.lora_shared_B_groups.values():
                        torch.testing.assert_close(parameter, source.get_submodule(name).lora_shared_B, rtol=0, atol=0)

    def test_common_initialization_resets_training_state_and_retains_policy(self):
        from utils import load_checkpoint
        cfg = make_stage_config(["B", "P", "B", "B"])
        source = build_stage_model(cfg)
        target = build_stage_model(cfg)
        optimizer = build_optimizer(cfg, target)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = os.path.join(folder, "common.pth")
            torch.save({"model": source.state_dict(), "optimizer": {"invalid": True},
                        "lr_scheduler": {}, "epoch": 99,
                        "extra_state": {"other_experiment": True}}, path)
            cfg.TRAIN.START_EPOCH = 0
            extra = {}
            info = {}
            scheduler = mock.Mock()
            scaler = mock.Mock()
            load_checkpoint(cfg, target, optimizer, scheduler, scaler, logging.getLogger("test"),
                            initialization_path=path, extra_state=extra, load_info=info)
            self.assertEqual(cfg.TRAIN.START_EPOCH, 0)
            self.assertFalse(info["restored_train_state"])
            self.assertFalse(optimizer.state)
            self.assertEqual(extra, {})
            scheduler.load_state_dict.assert_not_called()
            scaler.load_state_dict.assert_not_called()
            self.assertTrue(all(not p.requires_grad for p in disabled_adapter_parameters(target)))
            for module in target.iter_mtlora_modules(1):
                self.assertFalse(module.shared_lora_enabled)
            x = torch.randn(1, 3, 64, 64)
            torch.testing.assert_close(target(x, task="semseg"), source(x, task="semseg"), rtol=0, atol=0)
            cfg.MODEL.INIT_CHECKPOINT_SHA256 = "invalid"
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_checkpoint(cfg, target, optimizer, scheduler, scaler, logging.getLogger("test"), initialization_path=path)

    def test_evaluation_and_parameter_summary_use_training_policy(self):
        from experiment_utils import build_model_for_experiment
        from run_mtlora_parameter_summary import build_summary_payload
        cfg = make_stage_config(["G", "P", "B", "B"])
        with redirect_stdout(io.StringIO()):
            model = build_model_for_experiment(cfg, torch.device("cpu"))
        expected = adapter_parameter_summary(model)
        self.assertEqual(expected["stages"][0]["prompt_tokens"], 0)
        self.assertFalse(expected["stages"][1]["group_lora_enabled"])
        self.assertLess(expected["enabled_adapter_parameters"], expected["stored_adapter_parameters"])
        payload = build_summary_payload(SimpleNamespace(checkpoint="model.pth"), cfg, model)
        self.assertEqual(payload["adapter_summary"], expected)
        self.assertEqual(payload["stage_ablation"], stage_ablation_metadata(cfg))
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text()
        self.assertLess(source.index("prepare_trainable_parameters(model, config)"), source.index("optimizer = build_optimizer(config, model)"))
        self.assertLess(source.index("prepare_trainable_parameters(model, config)"), source.index("validate(config, data_loader_val, model, 0)"))

    def test_main_eval_and_initial_validation_apply_policy_before_optimizer(self):
        # Plotting is outside this setup/validation check and absent in the CPU test env.
        with mock.patch.dict("sys.modules", {"matplotlib": mock.Mock(), "matplotlib.pyplot": mock.Mock()}):
            import main as entry
        for eval_mode in (True, False):
            cfg = make_stage_config(["G", "P", "B", "B"])
            cfg.EVAL_MODE = eval_mode
            cfg.TRAIN.EPOCHS = 0  # Exercise setup and validation, with no training steps.
            cfg.TRAIN.ENABLE_CONFLICT_RATIO = False
            cfg.MODEL.RESUME_BACKBONE = False
            with redirect_stdout(io.StringIO()):
                model = build_model(cfg)
            observed_optimizers = []
            def optimizer_factory(config, module):
                self.assertTrue(all(not p.requires_grad for p in disabled_adapter_parameters(module)))
                optimizer = build_optimizer(config, module)
                observed_optimizers.append(optimizer)
                return optimizer
            def validation(config, loader, module, epoch):
                self.assertTrue(observed_optimizers)
                self.assertTrue(all(not p.requires_grad for p in disabled_adapter_parameters(module)))
                self.assertFalse(module.stage_prompt_enabled[0])
                self.assertTrue(all(not m.shared_lora_enabled for m in module.iter_mtlora_modules(1)))
                return 0.0, 0.0, 0.0
            with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                cfg.MODEL.INIT_CHECKPOINT = os.path.join(folder, "common.pth")
                torch.save({"model": model.state_dict()}, cfg.MODEL.INIT_CHECKPOINT)
                with mock.patch.object(entry, "logger", mock.Mock(), create=True), \
                     mock.patch.object(entry, "wandb_available", False), \
                     mock.patch.object(entry, "build_loader", return_value=(None, None, [None], [None], None)), \
                     mock.patch.object(entry, "build_model", return_value=model), \
                     mock.patch.object(model, "cuda", return_value=model), \
                     mock.patch.object(entry, "get_model_complexity_info", return_value=("0", "0")), \
                     mock.patch.object(entry, "build_optimizer", side_effect=optimizer_factory), \
                     mock.patch.object(entry, "build_scheduler", return_value=mock.Mock()), \
                     mock.patch.object(entry, "NativeScalerWithGradNormCount", return_value=mock.Mock()), \
                     mock.patch.object(entry.dist, "get_rank", return_value=0), \
                     mock.patch.object(entry, "validate", side_effect=validation) as validate:
                    entry.main(cfg)
                    self.assertEqual(validate.call_count, 1 if eval_mode else 2)

    def test_experiment_configs_share_formal_protocol(self):
        configs = []
        folder = Path(__file__).resolve().parents[1] / "configs/stage_ablation/swin_t"
        for name, modes in VARIANTS.items():
            args = make_args(str(folder / f"{name}.yaml"), tasks=None)
            args.pascal = None
            with redirect_stdout(io.StringIO()):
                cfg = get_config(args)
            self.assertEqual(cfg.MODEL.STAGE_ABLATION.MODES, modes)
            self.assertEqual(cfg.MODEL.PROMPT.NUM_TOKENS, 50)
            self.assertEqual(cfg.MODEL.MTLORA.AGMTLORA_GROUP_RANKS, [[16] * 4, [48] * 4])
            self.assertEqual(cfg.TASKS, ["semseg", "normals", "sal", "human_parts"])
            self.assertEqual(cfg.DATA.DBNAME, "PASCALContext")
            self.assertEqual(cfg.TRAIN.BASE_LR, 0.001)
            self.assertEqual(cfg.EVAL_FREQ, 5)
            cfg.defrost()
            cfg.MODEL.NAME = "same"
            cfg.OUTPUT = "same"
            cfg.MODEL.STAGE_ABLATION.MODES = ["B"] * 4
            for key in ("AFFINITY_SAVE_PATH", "GROUPING_SAVE_PATH", "META_SPLIT_SAVE_PATH"):
                cfg.MODEL.AGMTLORA[key] = "same"
            configs.append(cfg.dump())
        self.assertEqual(len(set(configs)), 1)


if __name__ == "__main__":
    unittest.main()
