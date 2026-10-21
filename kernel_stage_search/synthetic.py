"""Explicit CPU integration fixture; never supplies data to a research run."""

from pathlib import Path
import tempfile

import torch

from .cli import load_config
from .protocol import PROJECT_ROOT, TASKS, content_hash, make_manifest, resolved_identity, file_hash


class SyntheticDataset:
    def __init__(self, count=12, size=32):
        self.im_ids = [f"synthetic-{i:03d}" for i in range(count)]
        self.size = size

    def __getitem__(self, index):
        y, x = torch.meshgrid(torch.arange(self.size), torch.arange(self.size), indexing="ij")
        phase = index*0.31
        image = torch.stack((torch.sin(x*.2+phase), torch.cos(y*.2+phase), torch.sin((x+y)*.1+phase)))
        labels = ((x//4+y//4+index) % 3).unsqueeze(0)
        normals = torch.stack((torch.sin(x*.2+phase), torch.cos(y*.2+phase), torch.ones_like(x).float()))
        normals = normals/normals.norm(dim=0, keepdim=True)
        parts = labels.clone()
        if index % 4 == 0:
            parts.fill_(255)
        return {"image": image, "semseg": labels, "human_parts": parts, "sal": (labels % 2),
                "normals": normals, "meta": {"image": self.im_ids[index]}}


def tiny_config():
    config = load_config(PROJECT_ROOT/"configs/stage_ablation/swin_t/resolved_baseline.yaml")
    config.defrost()
    config.DATA.IMG_SIZE = [32, 32]
    config.TASKS_CONFIG.TRAIN.SCALE = (32, 32)
    config.TASKS_CONFIG.TEST.SCALE = (32, 32)
    config.MODEL.SWIN.EMBED_DIM = 8
    config.MODEL.SWIN.DEPTHS = [1, 1, 1, 1]
    config.MODEL.SWIN.NUM_HEADS = [1, 2, 4, 8]
    config.MODEL.SWIN.WINDOW_SIZE = 2
    config.MODEL.SWIN.MLP_RATIO = 2.
    config.MODEL.DROP_PATH_RATE = 0.
    config.MODEL.DROP_RATE = 0.
    config.MODEL.PROMPT.NUM_TOKENS = 2
    config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS = [[2]*4, [3]*4]
    config.MODEL.AGMTLORA.GROUP_SHARED_RANKS = [[2]*4, [3]*4]
    config.MODEL.AGMTLORA.RESOLVED_GROUP_SHARED_RANKS = [[2]*4, [3]*4]
    config.MODEL.DECODER_CHANNELS = [8, 16, 32, 32]
    config.MODEL.KERNEL_SEARCH.FORWARD_BATCH_SIZE = 2
    config.TRAIN.USE_CHECKPOINT = False
    config.freeze()
    return config


def _check_synthetic(device="cpu", output_dir=None):
    from models import build_model, build_mtl_model
    from .data import build_sample_plan, verify_sample_plan
    from .calibration import calibrate_model, validate_evaluation_labels
    from .checkpoint import state_digest
    from .policy import temporary_stage_modes
    from .runner import FrozenScorer, run_search
    from .export import load_completed_search
    torch.manual_seed(0)
    config = tiny_config()
    model = build_mtl_model(build_model(config), config).to(device).eval().requires_grad_(False)
    # Exercise nonzero adapters (standard initialization zeros B matrices).
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_shared_B" in name:
                parameter.normal_(std=.02)
    dataset = SyntheticDataset()
    calibration_plan = build_sample_plan(dataset, dataset.im_ids[:8], 2, 4, 8, 16)
    sample_plan = build_sample_plan(dataset, dataset.im_ids[8:], 2, 2, 4, 16)
    verify_sample_plan(sample_plan, dataset)
    before = state_digest(model)
    calibration = calibrate_model(model, config, dataset, calibration_plan)
    validate_evaluation_labels(sample_plan, calibration)
    def run(root):
        root.mkdir(parents=True, exist_ok=True)
        c0 = root/"synthetic_c0.pth"
        torch.save({"model": model.state_dict()}, c0)
        baseline = config.clone()
        baseline.defrost()
        baseline.BASE = ['']
        baseline.MODEL.INIT_CHECKPOINT = str(c0)
        baseline.MODEL.INIT_CHECKPOINT_SHA256 = file_hash(c0)
        baseline.freeze()
        baseline_path = root/"synthetic_baseline.yaml"
        baseline_path.write_text(baseline.dump(), encoding="utf-8")
        protocol = make_manifest("scoring", {"synthetic": True, "settings": {"SEARCH_BUDGET": 12},
            "mb_hash": before, "c0_hash": file_hash(c0), "identity": resolved_identity(baseline),
            "sample_plan_hash": sample_plan["sample_plan_id"],
            "calibration_hash": content_hash(calibration["sigma_features_squared"])})
        scorer = FrozenScorer(model, config, dataset, sample_plan, calibration, protocol, root/"cache")
        true = run_search(protocol, root/"true", "true_labels", scorer)
        observed = scorer.cost_summary()["forward_batches_started"]
        assert run_search(protocol, root/"true", "true_labels", scorer) == true
        assert observed == scorer.cost_summary()["forward_batches_started"]
        shuffled = run_search(protocol, root/"shuffled", "shuffled", scorer)
        load_completed_search(root/"true", "true_labels")
        load_completed_search(root/"shuffled", "shuffled")
        from .export import export_formal_configs
        exported = export_formal_configs(root/"true", root/"shuffled", root/"formal",
                                         baseline_config=baseline, synthetic_baseline_path=baseline_path)
        assert state_digest(model) == before
        return {"synthetic": True, "true_winner": true["modes"], "shuffled_winner": shuffled["modes"],
                "observations_per_branch": 12, "state_unchanged": True,
                "exported_unique_runs": len(exported["runs"]),
                "cache_hits": scorer.session_hits, "forward_batches": scorer.cost_summary()["forward_batches_started"]}
    if output_dir:
        return run(Path(output_dir).resolve())
    with tempfile.TemporaryDirectory(prefix="kernel-stage-synthetic-") as directory:
        return run(Path(directory))


def check_synthetic(device="cpu", output_dir=None):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _check_synthetic(device, output_dir)
    finally:
        torch.set_num_threads(previous)


def check_real_structure(device="cpu"):
    from models import build_model
    from .policy import temporary_stage_modes
    config = load_config(PROJECT_ROOT/"configs/stage_ablation/swin_t/resolved_baseline.yaml")
    model = build_model(config).to(device).eval().requires_grad_(False)
    image = torch.zeros(1, 3, 448, 448, device=device)
    for modes in ("BBBB", "PPPP", "GGGG", "PBGG", "GBPG"):
        with temporary_stage_modes(model, config, modes), torch.inference_mode():
            for task in TASKS:
                features = model(image, task=task, return_stages=True)
                assert [tuple(layer.output_resolution) for layer in model.layers] == [(56, 56), (28, 28), (14, 14), (14, 14)]
                assert [value.shape[-1] for value in features] == [192, 384, 768, 768]
                assert [value.shape[1] for value in features] == [3136, 784, 196, 196]
    return {"structural_smoke": True, "input_size": 448, "modes": ["BBBB", "PPPP", "GGGG", "PBGG", "GBPG"], "tasks": list(TASKS)}
