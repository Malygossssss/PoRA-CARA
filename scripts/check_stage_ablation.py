"""Dataset-free checks of all nine formal configs; never runs an optimizer step."""

import argparse
import gc
import io
import json
import logging
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config
from experiment_utils import build_model_for_experiment, set_random_seed
from optimizer import build_optimizer
from stage_ablation import adapter_parameter_summary, disabled_adapter_parameters, stage_ablation_metadata
from utils import load_checkpoint, save_checkpoint


class EmptyScaler:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("This smoke test only uses an empty CPU scaler state.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="output/stage_ablation_smoke.json")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    logger = logging.getLogger("stage_ablation_smoke")
    logger.addHandler(logging.NullHandler())
    variants = ["baseline"] + [f"S{i}-{mode}" for mode in ("P", "G") for i in range(1, 5)]
    results = []
    for variant in variants:
        options = SimpleNamespace(cfg=str(ROOT / "configs/stage_ablation/swin_t" / f"{variant}.yaml"),
                                  opts=None, local_rank=0)
        with redirect_stdout(io.StringIO()):
            config = get_config(options)
            set_random_seed(config.SEED, True)
            model = build_model_for_experiment(config, torch.device(args.device)).eval()
            optimizer = build_optimizer(config, model)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            scaler = EmptyScaler()
            load_checkpoint(config, model, optimizer, scheduler, scaler, logger,
                            initialization_path=config.MODEL.INIT_CHECKPOINT)
        x = torch.randn(1, 3, *config.DATA.IMG_SIZE, device=args.device)
        expected_tokens = [resolution ** 2 for resolution in model.input_res]
        feature_shapes = []
        with torch.no_grad():
            outputs = model(x)
        for task, output in outputs.items():
            assert list(output.shape) == [1, config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT[task], *config.DATA.IMG_SIZE]
            assert torch.isfinite(output).all(), (variant, task)
        if not args.forward_only:
            for task in config.TASKS:
                features = model.backbone(x, task=task, return_stages=True)
                assert [feature.shape[1] for feature in features] == expected_tokens
                feature_shapes.append([list(feature.shape) for feature in features])
                images = model.downsampler[task](features)
                # Same decoder inputs as MultiTaskSwin, one task at a time to bound CPU memory.
                logits = model.decoders.decoders[task](images)
                logits.square().mean().backward()
                del features, images, logits
            assert all(parameter.grad is None for parameter in disabled_adapter_parameters(model))
            assert model.backbone.patch_embed.proj.weight.grad.abs().sum() > 0
            assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                       for name, parameter in model.named_parameters()
                       if parameter.requires_grad and "lora_" in name)
        model.zero_grad(set_to_none=True)
        summary = adapter_parameter_summary(model)
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            config.defrost()
            config.OUTPUT = folder
            config.freeze()
            path = save_checkpoint(config, 0, model, 0.0, optimizer, scheduler, scaler, logger)
            config.defrost()
            config.MODEL.RESUME = path
            config.freeze()
            with torch.no_grad():
                model.backbone.patch_embed.proj.weight.add_(1.0)
            with redirect_stdout(io.StringIO()):
                load_checkpoint(config, model, optimizer, scheduler, scaler, logger)
            with torch.no_grad():
                restored_outputs = model(x)
            for task in config.TASKS:
                torch.testing.assert_close(restored_outputs[task], outputs[task], rtol=0, atol=0)
            assert adapter_parameter_summary(model) == summary
        row = {"variant": variant, "identity": stage_ablation_metadata(config), "parameters": summary,
               "feature_shapes": feature_shapes, "output_shapes": {t: list(v.shape) for t, v in outputs.items()},
               "forward": "passed", "backward": "skipped" if args.forward_only else "passed", "reload": "passed"}
        results.append(row)
        print(json.dumps({"variant": variant, **summary}, ensure_ascii=False), flush=True)
        del model, optimizer, scheduler, x, outputs, restored_outputs
        gc.collect()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
