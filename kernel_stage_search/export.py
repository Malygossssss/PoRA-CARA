"""Lock observed winners and export paired, fresh-from-C0 formal runs."""

from pathlib import Path
import copy

import yaml

from .protocol import (PROJECT_ROOT, content_hash, file_hash, read_json, read_manifest,
                       resolved_identity, write_json, validate_scoring_protocol)
from .runner import _checked_state
from .surrogate import select_winner, validate_observations


def load_completed_search(directory, label_mode):
    root = Path(directory)
    protocol = read_manifest(root/"resolved_protocol.json", "scoring")
    validate_scoring_protocol(protocol)
    state = _checked_state(root/"search_state.json")
    winner = read_json(root/"winner.json")
    identity = {"protocol_id": protocol["protocol_id"], "label_mode": label_mode,
                "budget": protocol["semantic"]["settings"]["SEARCH_BUDGET"]}
    validate_observations(state["observations"], identity["budget"])
    if state["identity"] != identity or state["pending"] is not None or len(state["observations"]) != identity["budget"]:
        raise ValueError("Export requires a complete search budget with no pending candidate")
    if winner.get("winner_hash") != content_hash({k: v for k, v in winner.items() if k != "winner_hash"}):
        raise ValueError("Winner record hash mismatch")
    best = select_winner(state["observations"])
    if (not winner.get("budget_complete") or any(winner.get(key) != value for key, value in identity.items())
            or any(winner.get(key) != best[key] for key in ("modes", "J", "score_hash"))
            or winner["observations_hash"] != content_hash(state["observations"])):
        raise ValueError("Winner must be max(actual observed J) from this completed search")
    for observation in state["observations"]:
        path = root/"scores"/(observation["modes"]+".json")
        if file_hash(path) != observation["score_hash"]:
            raise ValueError("Observed score hash mismatch")
    return protocol, winner


def export_formal_configs(true_run, shuffled_run, output_dir, baseline_config=None, synthetic_baseline_path=None):
    from .cli import load_config
    true_protocol, true_winner = load_completed_search(true_run, "true_labels")
    shuffled_protocol, shuffled_winner = load_completed_search(shuffled_run, "shuffled")
    if true_protocol["protocol_id"] != shuffled_protocol["protocol_id"]:
        raise ValueError("Paired controls must use the same complete scoring protocol")
    if true_protocol["semantic"].get("synthetic") and synthetic_baseline_path is None:
        raise ValueError("Synthetic observations cannot export research training configurations")
    baseline_path = PROJECT_ROOT/"configs/stage_ablation/swin_t/resolved_baseline.yaml"
    if synthetic_baseline_path is not None:
        if not true_protocol["semantic"].get("synthetic"):
            raise ValueError("An alternate fixture baseline is only allowed for explicit synthetic checks")
        baseline_path = Path(synthetic_baseline_path).resolve()
    baseline = baseline_config if baseline_config is not None else load_config(baseline_path)
    semantic = true_protocol["semantic"]
    if semantic.get("identity") != resolved_identity(baseline):
        raise ValueError("Formal baseline model/training identity differs from search protocol")
    c0 = Path(baseline.MODEL.INIT_CHECKPOINT)
    if not c0.is_absolute():
        c0 = PROJECT_ROOT/c0
    if file_hash(c0) != semantic["c0_hash"]:
        raise ValueError("Formal C0 checkpoint hash drift")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    groups = {"BBBB": "BBBB", "true_winner": true_winner["modes"], "shuffled_winner": shuffled_winner["modes"]}
    runs = []
    for modes in dict.fromkeys(groups.values()):
        for seed in (0, 1, 2):
            run_id = f"{modes}_seed{seed}"
            path = root/(run_id+".yaml")
            config = {"BASE": [str(baseline_path.resolve())], "MODEL": {
                "NAME": "swin_t_kernel_formal_"+run_id, "INIT_CHECKPOINT": str(c0.resolve()),
                "INIT_CHECKPOINT_SHA256": semantic["c0_hash"], "RESUME": "",
                "AGMTLORA": {"GROUPING_JSON": str(Path(baseline.MODEL.AGMTLORA.GROUPING_JSON).resolve())},
                "STAGE_ABLATION": {"MODES": list(modes)},
                "KERNEL_SEARCH": {"ENABLED": False, "PHASE": "off"}},
                "TRAIN": {"AUTO_RESUME": False, "START_EPOCH": 0, "EPOCHS": 300},
                "DATA": {"DATA_PATH": str((PROJECT_ROOT/baseline.DATA.DATA_PATH).resolve()),
                         "PASCAL": str((PROJECT_ROOT/baseline.DATA.DATA_PATH).resolve())},
                "SEED": seed, "OUTPUT": str(root/"training")}
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            # Resolve the exported file, including grouping and all inherited ranks.
            parsed = load_config(path)
            expected_identity = copy.deepcopy(semantic["identity"])
            expected_identity["training"]["seed"] = seed
            if (list(parsed.MODEL.STAGE_ABLATION.MODES) != list(modes) or parsed.TRAIN.LR_SCALED
                    or parsed.MODEL.RESUME or parsed.MODEL.KERNEL_SEARCH.ENABLED
                    or parsed.MODEL.INIT_CHECKPOINT_SHA256 != semantic["c0_hash"]
                    or resolved_identity(parsed) != expected_identity):
                raise ValueError("Exported training configuration failed validation")
            runs.append({"run_id": run_id, "modes": modes, "seed": seed, "config": str(path),
                         "config_hash": file_hash(path), "groups": [g for g, value in groups.items() if value == modes]})
    manifest = {"schema": 1, "protocol_id": true_protocol["protocol_id"], "c0_hash": semantic["c0_hash"],
                "synthetic": bool(semantic.get("synthetic", False)),
                "groups": groups, "runs": runs, "seeds": [0, 1, 2], "endpoint_epoch": 299,
                "scoring_protocol_path": str((Path(true_run)/"resolved_protocol.json").resolve()),
                "scoring_protocol_hash": file_hash(Path(true_run)/"resolved_protocol.json"),
                "initialization_lineage": semantic.get("initialization_lineage", {"status": "synthetic_or_unknown"}),
                "true_winner": true_winner, "shuffled_winner": shuffled_winner,
                "search_costs": {"true": read_json(Path(true_run)/"cost_summary.json"),
                                 "shuffled": read_json(Path(shuffled_run)/"cost_summary.json")},
                "paired_rng": "same C0, seed, loader and training settings; GPU bitwise identity is not guaranteed",
                "initialization_history": "C0 provenance retained; earlier unknown data history remains unknown"}
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(root/"formal_runs.json", manifest)
    return manifest
