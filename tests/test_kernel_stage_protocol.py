import copy
import tempfile
import unittest
from pathlib import Path

from kernel_stage_search.protocol import make_manifest, read_manifest, write_json, file_hash


class ProtocolTests(unittest.TestCase):
    def test_resolved_config_preserves_fixed_identity_and_rejects_drift(self):
        from kernel_stage_search.cli import load_config
        from kernel_stage_search.protocol import PROJECT_ROOT, validate_config, resolved_identity
        config = load_config(PROJECT_ROOT/"configs/kernel_stage_search/swin_t_pascal.yaml")
        self.assertEqual(config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS, [[16]*4, [48]*4])
        with tempfile.TemporaryDirectory() as root:
            snapshot = config.clone()
            snapshot.defrost()
            snapshot.BASE = ['']
            snapshot.freeze()
            path = Path(root)/"snapshot.yaml"
            path.write_text(snapshot.dump(), encoding="utf-8")
            self.assertEqual(resolved_identity(config), resolved_identity(load_config(path)))
        for key, value in (("SEARCH_BUDGET", 2), ("SOURCE_META_SPLIT", ""), ("PHASE", "off")):
            changed = config.clone()
            changed.defrost()
            changed.MODEL.KERNEL_SEARCH[key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)
        changed = config.clone()
        changed.defrost()
        changed.TASKS.reverse()
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_identity_ignores_output_location_but_binds_inputs(self):
        first = make_manifest("base", {"input_hash": "abc", "seed": 0}, {"output": "one"})
        second = make_manifest("base", first["semantic"], {"output": "two"})
        self.assertEqual(first["protocol_id"], second["protocol_id"])
        second["semantic"]["seed"] = 1
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "protocol.json"
            write_json(path, second)
            with self.assertRaisesRegex(ValueError, "hash"):
                read_manifest(path)

    def test_manifest_roundtrip_and_content_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "protocol.json"
            manifest = make_manifest("base", {"ranks": [[16]*4, [48]*4]})
            write_json(path, manifest)
            self.assertEqual(read_manifest(path), manifest)
            before = file_hash(path)
            write_json(path, {"changed": True})
            self.assertNotEqual(before, file_hash(path))

    def test_tensor_serialization_hash_ignores_output_filename(self):
        import torch
        from kernel_stage_search.calibration import atomic_torch_save
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root)/"first.pt", Path(root)/"another"/"second.pt"
            payload = {"stats": torch.arange(8).reshape(2, 4).float()}
            atomic_torch_save(first, payload)
            atomic_torch_save(second, payload)
            self.assertEqual(file_hash(first), file_hash(second))
