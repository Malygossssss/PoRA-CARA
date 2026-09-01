import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from yacs.config import CfgNode as CN

from config import get_config
from models.build import build_model
from models.lora import MTLoRALinear, mark_only_lora_as_trainable, mark_prompt_as_trainable
from models.swin_transformer_vpt import PromptedSwinTransformer, PromptedWindowAttention


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


if __name__ == "__main__":
    unittest.main()
