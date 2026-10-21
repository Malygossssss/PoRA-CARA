import unittest

import torch

from models import build_model
from kernel_stage_search.checkpoint import state_digest, load_strict_state
from kernel_stage_search.policy import temporary_stage_modes
from test_mtlora_prompt import make_stage_config


class PolicyTests(unittest.TestCase):
    def test_full_switch_restores_flags_outputs_and_tensors(self):
        for split_qkv in (False, True):
            config = make_stage_config("BBBB", split_qkv=split_qkv)
            config.DATA.IMG_SIZE = [32, 32]
            config.MODEL.SWIN.DEPTHS = [1, 1, 1, 1]
            model = build_model(config).eval().requires_grad_(False)
            x = torch.randn(1, 3, 32, 32)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            flags = [(m, m.shared_lora_enabled) for m in model.modules() if hasattr(m, "shared_lora_enabled")]
            with torch.inference_mode():
                original = model(x, task="semseg", return_stages=True)
                for modes in ("GGGG", "PBGG", "GBPG", "BBBB"):
                    with temporary_stage_modes(model, config, modes):
                        features = model(x, task="semseg", return_stages=True)
                        self.assertEqual([f.shape[1] for f in features], [16, 4, 1, 1])
                    for name, tensor in model.state_dict().items():
                        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
                with self.assertRaises(RuntimeError):
                    with temporary_stage_modes(model, config, "PPPP"):
                        raise RuntimeError("test restoration")
                restored = model(x, task="semseg", return_stages=True)
                for a, b in zip(original, restored):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual([(m, m.shared_lora_enabled) for m, _ in flags], flags)
            self.assertEqual(config.MODEL.STAGE_ABLATION.MODES, list("BBBB"))

    def test_p_disables_lora_and_restores_distinct_split_children(self):
        config = make_stage_config("BBBB", split_qkv=True)
        config.DATA.IMG_SIZE = [32, 32]
        config.MODEL.SWIN.DEPTHS = [1, 1, 1, 1]
        model = build_model(config).eval().requires_grad_(False)
        flags = [m for m in model.modules() if hasattr(m, "shared_lora_enabled")]
        flags[-1].shared_lora_enabled = False
        before = [m.shared_lora_enabled for m in flags]
        x = torch.randn(1, 3, 32, 32)
        with torch.inference_mode(), temporary_stage_modes(model, config, "PPPP"):
            original = model(x, task="semseg", return_stages=True)
            for name, value in model.named_parameters():
                if "lora_shared_" in name:
                    value.add_(torch.randn_like(value))
            perturbed = model(x, task="semseg", return_stages=True)
            for a, b in zip(original, perturbed):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual([m.shared_lora_enabled for m in flags], before)

    def test_strict_model_state_rejects_missing_parameter(self):
        model = torch.nn.BatchNorm1d(2)
        state = dict(model.state_dict())
        state.pop("weight")
        with self.assertRaisesRegex(ValueError, "key/shape"):
            load_strict_state(model, state)
