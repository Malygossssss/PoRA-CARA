import tempfile
import unittest
from pathlib import Path

from kernel_stage_search.reporting import summarize_records


class ReportingTests(unittest.TestCase):
    def test_export_parses_configs_and_deduplicates_shared_winner(self):
        import torch
        from kernel_stage_search.synthetic import tiny_config
        from kernel_stage_search.protocol import make_manifest, resolved_identity, file_hash
        from kernel_stage_search.runner import run_search
        from kernel_stage_search.export import export_formal_configs
        from kernel_stage_search.cli import load_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config()
            c0 = root/"synthetic_c0.pth"
            torch.save({"model": torch.nn.Linear(2, 2).state_dict()}, c0)
            config.defrost()
            config.BASE = ['']
            config.MODEL.INIT_CHECKPOINT = str(c0)
            config.MODEL.INIT_CHECKPOINT_SHA256 = file_hash(c0)
            config.freeze()
            baseline_path = root/"baseline.yaml"
            baseline_path.write_text(config.dump(), encoding="utf-8")
            protocol = make_manifest("scoring", {"synthetic": True, "settings": {"SEARCH_BUDGET": 3},
                         "identity": resolved_identity(config), "c0_hash": file_hash(c0)})
            for name, label in (("true", "true_labels"), ("shuffled", "shuffled")):
                run_search(protocol, root/name, label, lambda modes, labels: {"J": float(modes.count("G"))})
            manifest = export_formal_configs(root/"true", root/"shuffled", root/"formal", config, baseline_path)
            self.assertEqual(len(manifest["runs"]), 6)
            endpoints = []
            for run in manifest["runs"]:
                parsed = load_config(run["config"])
                self.assertEqual(parsed.SEED, run["seed"])
                self.assertEqual(parsed.MODEL.INIT_CHECKPOINT, str(c0))
                self.assertEqual(parsed.MODEL.MTLORA.AGMTLORA_GROUP_RANKS, [[2]*4, [3]*4])
                self.assertEqual(parsed.TRAIN.EPOCHS, 300)
                self.assertEqual(parsed.TRAIN.START_EPOCH, 0)
                self.assertFalse(parsed.MODEL.RESUME)
                self.assertFalse(parsed.TRAIN.AUTO_RESUME)
                self.assertFalse(parsed.MODEL.KERNEL_SEARCH.ENABLED)
                from stage_ablation import stage_ablation_metadata
                from kernel_stage_search.reporting import write_formal_endpoint
                checkpoint_path = Path(parsed.OUTPUT)/"ckpt_epoch_299.pth"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": torch.nn.Linear(2, 2).state_dict(), "epoch": 299,
                            "stage_ablation": stage_ablation_metadata(parsed)}, checkpoint_path)
                write_formal_endpoint(parsed, 299, checkpoint_path,
                    {"semseg": {"mIoU": .5}, "human_parts": {"mIoU": .4}, "sal": {"mIoU": .6}, "normals": {"mean": 20.}})
                endpoints.append(checkpoint_path.parent/"formal_endpoint_299.json")
            from kernel_stage_search.reporting import summarize_files
            summary = summarize_files(root/"formal"/"formal_runs.json", endpoints, root/"summary.json")
            self.assertEqual(summary["unique_training_runs"], 6)

    def test_fixed_endpoint_pairing_and_deduplicated_aliases(self):
        groups = {"BBBB": "BBBB", "true_winner": "PBGG", "shuffled_winner": "PBGG"}
        records = []
        for seed in range(3):
            for modes in ("BBBB", "PBGG"):
                records.append({"modes": modes, "seed": seed, "epoch": 299,
                                "checkpoint_hash": f"{modes}{seed}", "c0_hash": "c0",
                                "metrics": {"semseg_miou": .5+(modes != "BBBB")*.01, "human_miou": .4,
                                            "saliency_miou": .6, "normals_mean": 20.},
                                "units": {"semseg_miou": "fraction", "human_miou": "fraction",
                                          "saliency_miou": "fraction", "normals_mean": "degrees"}})
        summary = summarize_records(records, groups)
        self.assertAlmostEqual(summary["groups"]["true_winner"]["paired_differences"]["semseg_miou"]["mean"], .01)
        self.assertEqual(summary["groups"]["shuffled_winner"]["shared_modes_with"], ["true_winner"])
        records[0]["epoch"] = 298
        with self.assertRaisesRegex(ValueError, "299"):
            summarize_records(records, groups)
