import tempfile
import unittest
from pathlib import Path

from kernel_stage_search.protocol import make_manifest, read_json, write_json
from kernel_stage_search.runner import run_search


def synthetic_protocol(budget=12):
    return make_manifest("scoring", {"synthetic": True, "settings": {"SEARCH_BUDGET": budget}, "mb_hash": "mb",
                                     "c0_hash": "c0", "calibration_hash": "cal", "sample_plan_hash": "sample"})


class RunnerTests(unittest.TestCase):
    def test_actual_model_two_branches_share_kernels_and_export(self):
        from kernel_stage_search.synthetic import check_synthetic
        result = check_synthetic("cpu")
        self.assertTrue(result["state_unchanged"])
        self.assertEqual(result["observations_per_branch"], 12)
        self.assertGreater(result["cache_hits"], 0)
        self.assertLessEqual(result["forward_batches"], 42)  # initial three modes shared, at most 21 distinct.
        self.assertIn(result["exported_unique_runs"], (3, 6, 9))

    def test_budget_is_actual_calls_and_complete_resume_skips(self):
        calls = []
        def scorer(modes, label_mode):
            calls.append((modes, label_mode))
            return {"J": float(modes.count("G")), "tasks": {}, "batches": []}
        with tempfile.TemporaryDirectory() as root:
            protocol = synthetic_protocol()
            winner = run_search(protocol, root, "true_labels", scorer)
            self.assertEqual(len(calls), 12)
            self.assertTrue(winner["budget_complete"])
            self.assertEqual(winner["modes"], "GGGG")
            self.assertEqual(winner, run_search(protocol, root, "true_labels", scorer))
            self.assertEqual(len(calls), 12)
            with self.assertRaisesRegex(ValueError, "identity"):
                run_search(protocol, root, "shuffled", scorer)

    def test_failure_stops_without_zero_observation_and_resumes_pending(self):
        calls = []
        def failing(modes, labels):
            calls.append(modes)
            if modes == "PPPP":
                raise RuntimeError("interrupted")
            return {"J": 0.3}
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_search(synthetic_protocol(3), root, "true_labels", failing)
            state = read_json(Path(root)/"search_state.json")
            self.assertEqual(len(state["observations"]), 1)
            run_search(synthetic_protocol(3), root, "true_labels", lambda m, l: {"J": 0.2})
            self.assertEqual(len(read_json(Path(root)/"search_state.json")["observations"]), 3)

    def test_bad_score_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run_search(synthetic_protocol(3), root, "true_labels", lambda m, l: {"J": 0.2})
            path = Path(root)/"scores"/"BBBB.json"
            result = read_json(path)
            result["J"] = 99
            write_json(path, result)
            with self.assertRaisesRegex(ValueError, "hash"):
                run_search(synthetic_protocol(3), root, "true_labels", lambda m, l: {"J": 0.2})
