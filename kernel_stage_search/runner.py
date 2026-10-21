"""Serial frozen forwards, float32 feature-kernel cache and atomic GP recovery."""

import math
from pathlib import Path
import time

import torch

from .calibration import atomic_torch_save
from .checkpoint import state_digest
from .features import collect_batch_features, concatenate_features
from .kernels import gaussian_kernel, kernel_diagnostics, label_kernels, score_batch, validate_kernel
from .policy import temporary_stage_modes
from .protocol import TASKS, content_hash, file_hash, read_json, write_json, validate_scoring_protocol
from .surrogate import propose, select_winner, validate_observations


def _checked_state(path):
    state = read_json(path)
    if state.get("state_hash") != content_hash({k: v for k, v in state.items() if k != "state_hash"}):
        raise ValueError("Search state hash mismatch")
    return state


def _persist(root, state):
    state["state_hash"] = content_hash({k: v for k, v in state.items() if k != "state_hash"})
    # This one file is authoritative. JSONL files are regenerated views after recovery.
    write_json(root/"search_state.json", state)
    for name, entries in (("observations", state["observations"]), ("proposal_history", state["proposals"])):
        from .protocol import canonical_json
        path = root/(name+".jsonl")
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(canonical_json(entry)+"\n" for entry in entries), encoding="utf-8")
        temporary.replace(path)


def run_search(protocol, output_dir, label_mode, scorer):
    validate_scoring_protocol(protocol)
    if protocol.get("kind") != "scoring" or label_mode not in ("true_labels", "shuffled"):
        raise ValueError("Search requires a full scoring protocol and explicit label mode")
    identity = {key: protocol[key] for key in ("schema", "kind", "semantic")}
    if content_hash(identity) != protocol["protocol_id"]:
        raise ValueError("Scoring protocol hash mismatch")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    budget = protocol["semantic"]["settings"]["SEARCH_BUDGET"]
    identity = {"protocol_id": protocol["protocol_id"], "label_mode": label_mode, "budget": budget}
    state_path = root/"search_state.json"
    if state_path.exists():
        state = _checked_state(state_path)
        if state["identity"] != identity:
            raise ValueError("Search cache identity mismatch")
    else:
        if any((root/name).exists() for name in ("winner.json", "observations.jsonl", "scores")):
            raise ValueError("Orphaned search artifacts without authoritative state")
        state = {"schema": 1, "identity": identity, "observations": [], "proposals": [], "pending": None}
    validate_observations(state["observations"], budget)
    for observation in state["observations"]:
        path = root/"scores"/(observation["modes"]+".json")
        if not path.is_file() or file_hash(path) != observation["score_hash"]:
            raise ValueError("Completed score cache hash mismatch")
    write_json(root/"resolved_protocol.json", protocol)
    while True:
        proposal = propose(state["observations"], budget)
        if proposal is None:
            break
        if state["pending"] is not None and state["pending"] != proposal:
            raise ValueError("Pending proposal differs from deterministic GP recovery")
        if state["pending"] is None:
            state["proposals"].append({"observation_index": len(state["observations"]), **proposal})
        state["pending"] = proposal
        _persist(root, state)
        modes = proposal["modes"]
        score_path = root/"scores"/(modes+".json")
        score_identity = {**identity, "modes": modes}
        previous = read_json(score_path) if score_path.exists() else None
        if previous is not None and previous["identity"] != score_identity:
            raise ValueError("Score cache identity mismatch")
        if previous is not None and previous["status"] == "complete":
            if previous.get("result_hash") != content_hash({k: v for k, v in previous.items() if k != "result_hash"}):
                raise ValueError("Score cache result hash mismatch")
            result = previous
        else:
            write_json(score_path, {"identity": score_identity, "status": "pending"})
            started = time.perf_counter()
            try:
                result = scorer(modes, label_mode)
                if not math.isfinite(result["J"]):
                    raise ValueError("Non-finite actual J")
                result = {**result, "identity": score_identity, "status": "complete", "elapsed_seconds": time.perf_counter()-started}
                result["result_hash"] = content_hash(result)
                write_json(score_path, result)
            except BaseException as error:
                write_json(score_path, {"identity": score_identity, "status": "failed", "error": str(error),
                                        "elapsed_seconds": time.perf_counter()-started})
                raise
        state["observations"].append({"modes": modes, "J": result["J"], "score_hash": file_hash(score_path)})
        state["pending"] = None
        _persist(root, state)
    winner = select_winner(state["observations"])
    baseline = state["observations"][0]["J"]
    details = read_json(root/"scores"/(winner["modes"]+".json"))
    summary = {**identity, **winner, "J_BBBB": baseline, "delta_J": winner["J"]-baseline,
               "tasks": details.get("tasks", {}), "budget_complete": True,
               "observations_hash": content_hash(state["observations"]),
               "sources": {key: value for key, value in protocol["semantic"].items()
                           if key in ("c0_hash", "mb_hash", "calibration_hash", "sample_plan_hash")}}
    summary["winner_hash"] = content_hash(summary)
    write_json(root/"winner.json", summary)
    costs = scorer.cost_summary() if hasattr(scorer, "cost_summary") else {}
    write_json(root/"cost_summary.json", {"observed_configurations": len(state["observations"]), **costs})
    _persist(root, state)
    return summary


class FrozenScorer:
    def __init__(self, model, config, dataset, plan, calibration, protocol, cache_dir):
        self.model, self.config, self.dataset = model, config, dataset
        self.plan, self.calibration, self.protocol = plan, calibration, protocol
        self.root = Path(cache_dir)/protocol["protocol_id"]
        self.initial_digest = state_digest(model)
        self.session_hits, self.io_seconds = 0, 0.
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("M_B must be frozen in eval mode")

    def _kernels(self, modes, batch):
        # This identity deliberately excludes label mode and permutation.
        identity = {"protocol_id": self.protocol["protocol_id"], "modes": modes, "batch": batch["batch"],
                    "mb_hash": self.protocol["semantic"]["mb_hash"],
                    "sample_plan_hash": self.protocol["semantic"]["sample_plan_hash"],
                    "calibration_hash": self.protocol["semantic"]["calibration_hash"], "dtype": "float32"}
        path = self.root/modes/(f"batch_{batch['batch']:03d}.pt")
        metadata_path = path.with_suffix(".json")
        before_io = time.perf_counter()
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["identity"] != identity or not path.exists() or file_hash(path) != metadata["file_hash"]:
                raise ValueError("Feature-kernel cache hash/identity mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["identity"] != identity or set(payload["kernels"]) != set(TASKS):
                raise ValueError("Feature-kernel payload identity mismatch")
            for kernel in payload["kernels"].values():
                if kernel.dtype != torch.float32 or len(kernel) != self.plan["samples_per_task"]:
                    raise ValueError("Feature-kernel cache precision/size mismatch")
                validate_kernel(kernel)
            self.session_hits += 1
            self.io_seconds += time.perf_counter()-before_io
            return payload, metadata["file_hash"]
        event_path = self.root/"cost_events"/(str(time.time_ns())+".json")
        event = {"identity": identity, "status": "pending", "started_unix": time.time(), "image_count": len(batch["image_ids"])}
        write_json(event_path, event)
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            with temporary_stage_modes(self.model, self.config, modes):
                stages = collect_batch_features(self.model, self.dataset, batch,
                                                self.config.MODEL.KERNEL_SEARCH.FORWARD_BATCH_SIZE,
                                                self.calibration["stats"])
            features = concatenate_features(stages)
            kernels = {task: gaussian_kernel(features[task], self.calibration["sigma_features_squared"][task]) for task in TASKS}
            diagnostics = {task: kernel_diagnostics(kernels[task], features[task]) for task in TASKS}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            event.update(status="complete", forward_kernel_seconds=time.perf_counter()-start,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
            payload = {"identity": identity, "kernels": kernels, "diagnostics": diagnostics}
            write_started = time.perf_counter()
            atomic_torch_save(path, payload)
            checksum = file_hash(path)
            write_json(metadata_path, {"identity": identity, "file_hash": checksum})
            event["cache_bytes"] = path.stat().st_size
            event["cache_write_and_hash_seconds"] = time.perf_counter()-write_started
            write_json(event_path, event)
            return payload, checksum
        except BaseException as error:
            event.update(status="failed", elapsed_seconds=time.perf_counter()-start, error=str(error))
            write_json(event_path, event)
            raise

    @torch.inference_mode()
    def __call__(self, modes, label_mode):
        from stage_ablation import adapter_parameter_summary
        if state_digest(self.model) != self.initial_digest:
            raise ValueError("M_B weights/buffers changed before scoring")
        batches = []
        with temporary_stage_modes(self.model, self.config, modes):
            parameters = adapter_parameter_summary(self.model)
        for batch in self.plan["batches"]:
            payload, checksum = self._kernels(modes, batch)
            targets = label_kernels(batch, self.calibration["sigma_normals_squared"], label_mode)
            scored = score_batch(payload["kernels"], targets, self.config.MODEL.KERNEL_SEARCH.ENTROPY_MIN)
            scored.update(batch=batch["batch"], image_count=len(batch["image_ids"]),
                          kernel_hash=checksum, permutation_hash=content_hash({t: batch["tasks"][t]["permutation"] for t in TASKS}),
                          diagnostics=payload["diagnostics"],
                          label_distributions={t: batch["tasks"][t]["label_distribution"] for t in TASKS})
            batches.append(scored)
        final_digest = state_digest(self.model)
        if final_digest != self.initial_digest:
            raise ValueError("M_B weights/buffers changed during scoring")
        tasks = {task: sum(batch["tasks"][task]["ratio"] for batch in batches)/len(batches) for task in TASKS}
        return {"J": sum(batch["J"] for batch in batches)/len(batches), "tasks": tasks, "batches": batches,
                "parameters": parameters, "initial_state_hash": self.initial_digest, "final_state_hash": final_digest}

    def cost_summary(self):
        events = [read_json(path) for path in sorted((self.root/"cost_events").glob("*.json"))]
        one_time = {}
        for phase in ("preparation", "calibration"):
            path = self.protocol.get("paths", {}).get(phase+"_cost")
            one_time[phase] = read_json(path) if path and Path(path).exists() else {"status": "unknown_or_not_recorded"}
        return {"scope": "all branches sharing this protocol's feature-kernel cache", "events": events,
                "one_time_costs": one_time,
                "forward_batches_started": len(events), "session_cache_hits": self.session_hits,
                "session_cache_read_seconds": self.io_seconds,
                "total_cache_bytes": sum(path.stat().st_size for path in self.root.rglob("*.pt")),
                "forward_kernel_seconds": sum(event.get("forward_kernel_seconds", event.get("elapsed_seconds", 0.)) for event in events),
                "incomplete_cost_events": sum(event["status"] == "pending" for event in events)}
