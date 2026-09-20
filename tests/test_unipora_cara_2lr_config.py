import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "mtlora" / "tiny_448" / "pascal"
BASE_CONFIG_NAME = "unipora_cara_tiny_448_r64_prom50_global_group_proxy.yaml"
DOUBLE_LR_CONFIG_NAME = "unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr.yaml"


class UniPoRACARA2LRConfigTests(unittest.TestCase):
    def test_double_lr_config_inherits_existing_pipeline_config(self):
        config_path = CONFIG_DIR / DOUBLE_LR_CONFIG_NAME

        self.assertTrue(config_path.is_file(), f"Missing 2lr config: {config_path}")
        source = config_path.read_text(encoding="utf-8")
        self.assertRegex(
            source,
            rf"(?m)^BASE:\s*\n\s*-\s*{re.escape(BASE_CONFIG_NAME)}\s*$",
        )
        self.assertRegex(
            source,
            r"(?m)^\s*NAME:\s*unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr\s*$",
        )

    def test_double_lr_config_contains_exact_unipora_learning_rates(self):
        source = (CONFIG_DIR / DOUBLE_LR_CONFIG_NAME).read_text(encoding="utf-8")

        self.assertRegex(source, r"(?m)^\s*BASE_LR:\s*1\.0e-3\s*$")
        self.assertRegex(source, r"(?m)^\s*WARMUP_LR:\s*1\.0e-6\s*$")
        self.assertRegex(source, r"(?m)^\s*MIN_LR:\s*1\.0e-5\s*$")

    def test_stage1_and_resolved_config_propagate_the_runtime_base_lr(self):
        stage1_source = (PROJECT_ROOT / "ag_mtlora" / "stage1.py").read_text(encoding="utf-8")

        self.assertIn("optimizer = build_optimizer(config, model)", stage1_source)
        self.assertNotIn("build_scheduler", stage1_source)
        self.assertIn(
            "resolved_config.BASE = [os.path.abspath(base_cfg_path)]",
            stage1_source,
        )

    def test_formal_training_keeps_unipora_lr_scaling(self):
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn("linear_scaled_lr = config.TRAIN.BASE_LR * \\", main_source)
        self.assertIn(
            "config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0",
            main_source,
        )
        self.assertIn("config.TRAIN.BASE_LR = linear_scaled_lr", main_source)
        self.assertIn("config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr", main_source)
        self.assertIn("config.TRAIN.MIN_LR = linear_scaled_min_lr", main_source)


if __name__ == "__main__":
    unittest.main()
