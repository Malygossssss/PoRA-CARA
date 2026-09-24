"""Numerical contracts for the opt-in prompt rank residual experiment."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.lora import MTLoRALinear, PromptRankResidual
from models.build import build_model
from config import get_config
from tests.test_mtlora_prompt import make_args, make_model_config, make_rank_extract_yaml
from utils import _rank_extract_checkpoint_mode


class PromptRankResidualTests(unittest.TestCase):
    def test_experiment_configs_share_training_and_group_rank_budget(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs/mtlora/tiny_448/pascal"
        prefix = "unipora_cara_tiny_448_r64_prom50_"
        experiments = ("corrected_e0", "rank_extract_e4_last", "prompt_rank_residual_e5")
        with tempfile.TemporaryDirectory() as directory:
            grouping = Path(directory) / "groups.json"
            grouping.write_text(json.dumps({"groups": [["semseg", "sal"], ["normals", "human_parts"]]}))
            configs = []
            for experiment in experiments:
                args = make_args(str(config_dir / f"{prefix}{experiment}.yaml"),
                                 tasks="semseg,normals,sal,human_parts")
                args.opts = ["MODEL.AGMTLORA.GROUPING_JSON", str(grouping)]
                configs.append(get_config(args))
        reference = configs[0]
        self.assertFalse(reference.MODEL.MTLORA.RANK_EXTRACT.ENABLED)
        for config in configs[1:]:
            self.assertEqual(config.TRAIN, reference.TRAIN)
            self.assertEqual(config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS,
                             reference.MODEL.MTLORA.AGMTLORA_GROUP_RANKS)
            self.assertEqual(config.MODEL.MTLORA.RANK_EXTRACT.BLOCKS, "stage_last")
        self.assertEqual(configs[1].MODEL.MTLORA.RANK_EXTRACT.MODE, "gate")
        self.assertEqual(configs[2].MODEL.MTLORA.RANK_EXTRACT.MODE, "prompt_residual")

    def test_model_build_target_scope_and_initial_identity(self):
        config = make_model_config(ag_enabled=True)
        config.MODEL.SWIN.DEPTHS = [2, 2, 2, 2]
        config.MODEL.MTLORA.defrost()
        for ranks in config.MODEL.MTLORA.R_PER_TASK_LIST:
            ranks["semseg"] = ranks["sal"] = 0
        settings = config.MODEL.MTLORA.RANK_EXTRACT
        settings.set_new_allowed(True)
        settings.ENABLED = True
        settings.MODE = "prompt_residual"
        settings.BLOCKS = "stage_last"
        config.MODEL.MTLORA.freeze()
        model = build_model(config).eval()
        linears = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, MTLoRALinear) and module.rank_extract_enabled]
        self.assertEqual([name for name, _ in linears],
                         ["layers.2.blocks.1.mlp.fc1", "layers.3.blocks.1.mlp.fc1"])
        for _, linear in linears:
            for residual in linear.lora_rank_extractors.values():
                self.assertIsInstance(residual, PromptRankResidual)
                self.assertEqual(residual.writeback.weight.count_nonzero().item(), 0)
        # Use nonzero B so this comparison really exercises the adapters.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_shared_B_groups" in name:
                    parameter.normal_(std=0.01)
            images = torch.randn(2, 3, 32, 32)
            residual_features = model(images, task="semseg", return_stages=True)
            for _, linear in linears:
                linear.rank_extract_enabled = False
            baseline_features = model(images, task="semseg", return_stages=True)
        for baseline, actual in zip(baseline_features, residual_features):
            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_config_and_incompatible_gate_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            grouping = Path(directory) / "groups.json"
            grouping.write_text(json.dumps({"groups": [["semseg"], ["sal"]]}))
            config_path = Path(directory) / "config.yaml"
            content = make_rank_extract_yaml(str(grouping)).replace(
                "HIDDEN_DIM: 16", "HIDDEN_DIM: 16\n      MODE: prompt_residual\n      TEMPERATURE: 0.25")
            config_path.write_text(content)
            config = get_config(make_args(str(config_path)))
            self.assertEqual(config.MODEL.MTLORA.RANK_EXTRACT.MODE, "prompt_residual")
            config_path.write_text(content.replace("TEMPERATURE: 0.25", "TEMPERATURE: 0.0"))
            with self.assertRaisesRegex(ValueError, "TEMPERATURE"):
                get_config(make_args(str(config_path)))
        residual_state = self.make_layer().state_dict()
        mode, _ = _rank_extract_checkpoint_mode({}, residual_state)
        self.assertEqual(mode, "initialization")
        mode, _ = _rank_extract_checkpoint_mode(residual_state, residual_state)
        self.assertEqual(mode, "resume")
        gate_state = {"lora_rank_extractors.group_0.output.weight": torch.zeros(2, 2)}
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _rank_extract_checkpoint_mode(gate_state, residual_state)

    def test_identity_and_first_step_writeback_gradient(self):
        module = PromptRankResidual(3)
        z = torch.randn(2, 5, 3, requires_grad=True)
        prompts = torch.randn(2, 4, 3, requires_grad=True)
        result = module(z, prompts)
        torch.testing.assert_close(result, z, rtol=0, atol=0)
        result.square().sum().backward()
        self.assertGreater(module.writeback.weight.grad.abs().sum().item(), 0)
        # The zero writeback intentionally delays the prompt-side gradient.
        torch.testing.assert_close(prompts.grad, torch.zeros_like(prompts))

    def test_same_mean_prompts_can_produce_different_patch_updates(self):
        module = PromptRankResidual(2, residual_scale=1.0)
        with torch.no_grad():
            module.writeback.weight.copy_(torch.eye(2))
        z = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        first = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        second = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])
        torch.testing.assert_close(first.mean(1), second.mean(1))
        self.assertFalse(torch.allclose(module(z, first), module(z, second)))

    def test_can_write_to_zero_rank_coordinate(self):
        module = PromptRankResidual(2, residual_scale=1.0)
        with torch.no_grad():
            module.writeback.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
        z = torch.tensor([[[1.0, 0.0]]])
        prompts = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        self.assertGreater(module(z, prompts)[0, 0, 1].item(), 0.5)

    def test_prompt_order_invariance_and_uniform_retrieval_identity(self):
        module = PromptRankResidual(3)
        with torch.no_grad():
            module.writeback.weight.normal_()
        z = torch.randn(2, 5, 3)
        prompts = torch.randn(2, 4, 3)
        torch.testing.assert_close(module(z, prompts), module(z, prompts.flip(1)))
        # Repeated prompts carry no token-specific information after centering.
        repeated = prompts[:, :1].expand(-1, 4, -1)
        torch.testing.assert_close(module(z, repeated), z)
        zeros = torch.zeros_like(z)
        torch.testing.assert_close(module(zeros, prompts), zeros, atol=1e-6, rtol=0)

    def test_uncentered_control_retains_global_prompt_information(self):
        module = PromptRankResidual(2, residual_scale=1.0, center_values=False)
        with torch.no_grad():
            module.writeback.weight.copy_(torch.eye(2))
        z = torch.zeros(1, 3, 2)
        prompts = torch.tensor([[[1.0, -2.0], [1.0, -2.0]]])
        expected = prompts[:, :1].expand_as(z)
        torch.testing.assert_close(module(z, prompts), expected)

    def test_batch_independence(self):
        module = PromptRankResidual(3)
        with torch.no_grad():
            module.writeback.weight.normal_()
        z, prompts = torch.randn(2, 5, 3), torch.randn(2, 4, 3)
        expected = torch.cat([module(z[i:i+1], prompts[i:i+1]) for i in range(2)])
        torch.testing.assert_close(module(z, prompts), expected)

    def make_layer(self, dropout=0.0):
        layer = MTLoRALinear(
            4, 6, r={"group_0": 3, "group_1": 2},
            tasks=["semseg", "sal"],
            task_to_group={"semseg": "group_0", "sal": "group_1"},
            rank_extract_enabled=True, rank_extract_mode="prompt_residual",
            lora_dropout=dropout,
        )
        with torch.no_grad():
            for weight in layer.lora_shared_B_groups.values():
                weight.normal_()
        return layer

    def test_identity_preserves_baseline_and_dropout(self):
        layer = self.make_layer(dropout=0.4).train()
        x, task_x = torch.randn(2, 7, 4), torch.randn(2, 7, 4)
        layer.rank_extract_enabled = False
        torch.manual_seed(11)
        _, baseline = layer(x, {"semseg": task_x}, active_task="semseg")
        layer.rank_extract_enabled = True
        torch.manual_seed(11)
        _, updated = layer(x, {"semseg": task_x}, active_task="semseg", prompt_token_count=3)
        torch.testing.assert_close(updated["semseg"], baseline["semseg"], rtol=0, atol=0)

    def test_only_patch_residual_and_selected_group_are_updated(self):
        layer = self.make_layer()
        x = torch.randn(2, 7, 4)
        task_x = torch.randn(2, 7, 4, requires_grad=True)
        _, baseline = layer(x, {"semseg": task_x}, active_task="semseg", prompt_token_count=3)
        with torch.no_grad():
            layer.lora_rank_extractors["group_0"].writeback.weight.normal_()
        _, result = layer(x, {"semseg": task_x}, active_task="semseg", prompt_token_count=3)
        torch.testing.assert_close(result["semseg"][:, :3], baseline["semseg"][:, :3])
        self.assertFalse(torch.allclose(result["semseg"][:, 3:], baseline["semseg"][:, 3:]))
        result["semseg"][:, 3:].square().sum().backward()
        self.assertGreater(task_x.grad[:, :3].abs().sum().item(), 0)
        self.assertGreater(layer.lora_shared_A_groups["group_0"].grad.abs().sum().item(), 0)
        self.assertIsNone(layer.lora_shared_A_groups["group_1"].grad)
        self.assertIsNone(layer.lora_rank_extractors["group_1"].writeback.weight.grad)

    def test_autocast_finite_and_retains_rank_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                module = PromptRankResidual(4)
                with torch.no_grad():
                    module.writeback.weight.normal_()
                z = torch.randn(2, 6, 4).to(dtype).requires_grad_()
                prompts = torch.randn(2, 5, 4).to(dtype).requires_grad_()
                with torch.autocast("cpu", dtype=dtype):
                    result = module(z, prompts)
                    loss = result.float().square().mean()
                loss.backward()
                self.assertEqual(result.dtype, z.dtype)
                for value in (result, z.grad, prompts.grad, module.writeback.weight.grad):
                    self.assertTrue(torch.isfinite(value).all())

    def test_invalid_parameters_and_shapes(self):
        for kwargs in ({"rank": 0}, {"rank": 2, "temperature": 0},
                       {"rank": 2, "residual_scale": float("nan")},
                       {"rank": 2, "eps": 0}):
            with self.assertRaises(ValueError):
                PromptRankResidual(**kwargs)
        module = PromptRankResidual(2)
        for prompts in (torch.randn(1, 2), torch.randn(3, 2, 2), torch.randn(1, 0, 2)):
            with self.assertRaises(ValueError):
                module(torch.randn(1, 4, 2), prompts)


if __name__ == "__main__":
    unittest.main()
