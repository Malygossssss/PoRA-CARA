import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from kernel_stage_search.data import validate_split, valid_labels, build_sample_plan, verify_sample_plan


class TinyDataset:
    def __init__(self, reverse=False):
        self.im_ids = ["a", "b", "c", "d"][::(-1 if reverse else 1)]

    def __getitem__(self, index):
        name = self.im_ids[index]
        labels = torch.arange(16).reshape(1, 4, 4) % 2
        return {"image": torch.ones(3, 4, 4)*ord(name), "meta": {"image": name},
                "semseg": labels, "human_parts": labels, "sal": labels,
                "normals": torch.stack((labels[0].float(), torch.ones(4, 4), torch.zeros(4, 4)))}


class DataTests(unittest.TestCase):
    def test_split_is_strict(self):
        good = {"meta_train_ids": ["a", "b"], "meta_val_ids": ["c"]}
        self.assertEqual(validate_split(good, ["a", "b", "c"], ["v"])["meta_val_ids"], ["c"])
        for bad in ({"meta_train_ids": ["a", "a"], "meta_val_ids": ["c"]},
                    {"meta_train_ids": ["a", "c"], "meta_val_ids": ["c"]},
                    {"meta_train_ids": ["a"], "meta_val_ids": ["missing"]}):
            with self.assertRaises(ValueError):
                validate_split(bad, ["a", "b", "c"], ["v"])
        with self.assertRaises(ValueError):
            validate_split(good, ["a", "b", "c"], ["c"])

    def test_validity_and_normal_vector_units(self):
        values, mask = valid_labels("human_parts", torch.full((1, 2, 2), 255.))
        self.assertEqual(mask.sum(), 0)
        n = torch.tensor([[[1., 0., 255., float("nan")]], [[1., 0., 0., 0.]], [[0., 0., 0., 0.]]])
        values, mask = valid_labels("normals", n)
        self.assertEqual(mask.sum(), 1)
        self.assertAlmostEqual(float(np.linalg.norm(values[mask][0])), 1., places=6)
        with self.assertRaises(ValueError):
            valid_labels("sal", torch.tensor([[[0.2]]]))

    def test_plan_uses_ids_and_fixed_permutations(self):
        options = dict(num_batches=2, initial_images=2, max_images=4, samples=8, seed=0, shuffle_seed=1)
        plan = build_sample_plan(TinyDataset(), ["a", "b", "c", "d"], **options)
        reordered = build_sample_plan(TinyDataset(True), ["a", "b", "c", "d"], **options)
        self.assertEqual(plan, reordered)
        verify_sample_plan(plan, TinyDataset(True))
        normals = np.array(plan["batches"][0]["tasks"]["normals"]["labels"])
        perm = plan["batches"][0]["tasks"]["normals"]["permutation"]
        np.testing.assert_allclose(np.sort(normals[:, 0]), np.sort(normals[perm, 0]))

    def test_degenerate_labels_and_input_content_drift_fail(self):
        dataset = TinyDataset()
        plan = build_sample_plan(dataset, dataset.im_ids, 1, 2, 4, 8)
        class Changed(TinyDataset):
            def __getitem__(self, index):
                value = super().__getitem__(index)
                value["image"] += .01
                return value
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            verify_sample_plan(plan, Changed())
        class Constant(TinyDataset):
            def __getitem__(self, index):
                value = super().__getitem__(index)
                value["sal"].fill_(1)
                return value
        with self.assertRaisesRegex(ValueError, "degenerate"):
            build_sample_plan(Constant(), dataset.im_ids, 1, 2, 4, 8)
