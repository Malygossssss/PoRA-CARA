import unittest

import numpy as np
import torch

from kernel_stage_search.kernels import gaussian_kernel, equality_kernel, normalized_information, median_positive_distance


class KernelTests(unittest.TestCase):
    def test_constant_and_identical_label_kernels(self):
        y = equality_kernel(torch.tensor([0, 0, 1, 1]))
        constant = gaussian_kernel(torch.ones(4, 2), 1.)
        self.assertAlmostEqual(normalized_information(constant, y)["ratio"], 0., places=12)
        self.assertAlmostEqual(normalized_information(y, y)["ratio"], 1., places=12)
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            normalized_information(y, torch.ones(4, 4))

    def test_crossed_independence_reference_and_joint_permutation(self):
        kf = equality_kernel(torch.tensor([0, 0, 1, 1]))
        ky = equality_kernel(torch.tensor([0, 1, 0, 1]))
        result = normalized_information(kf, ky)
        self.assertAlmostEqual(result["i2"], 0.)
        perm = torch.tensor([3, 0, 2, 1])
        self.assertEqual(result, normalized_information(kf[perm][:, perm], ky[perm][:, perm]))
        for key, matrix in (("sx", kf/4), ("sy", ky/4), ("sxy", (kf*ky)/4)):
            reference = -torch.log2(torch.linalg.eigvalsh(matrix.double()).square().sum()).item()
            self.assertAlmostEqual(result[key], reference, places=12)

    def test_float32_distance_and_fixed_width(self):
        x = torch.tensor([[0., 0.], [1., 0.], [0., 2.]])
        width = median_positive_distance(x)
        self.assertEqual(width, 4.)
        k = gaussian_kernel(x, width)
        self.assertEqual(k.dtype, torch.float32)
        self.assertAlmostEqual(k[0, 1].item(), np.exp(-1/8), places=6)
        self.assertEqual(normalized_information(k, equality_kernel(torch.tensor([0, 1, 0])))["n"], 3)
        with self.assertRaises(ValueError):
            median_positive_distance(torch.zeros(3, 1))

    def test_malformed_kernels_fail(self):
        good = torch.eye(3)
        for bad in (torch.ones(2, 3), torch.full((3, 3), float("nan")), torch.zeros(3, 3)):
            with self.assertRaises(ValueError):
                normalized_information(bad, good)

    def test_training_only_streaming_calibration_and_shuffled_entropy(self):
        from kernel_stage_search.calibration import fit_calibration
        from kernel_stage_search.protocol import TASKS
        rng = torch.Generator().manual_seed(2)
        batches = []
        for _ in range(2):
            stages = {task: [torch.randn(8, stage+2, generator=rng) for stage in range(4)] for task in TASKS}
            normals = torch.randn(8, 3, generator=rng)
            normals /= normals.norm(dim=1, keepdim=True)
            batches.append((stages, normals))
        calibration = fit_calibration(batches, bandwidth_samples=8)
        for task in TASKS:
            reference = torch.cat([batch[0][task][0] for batch in batches])
            torch.testing.assert_close(calibration["stats"][task][0]["mean"], reference.mean(dim=0))
            torch.testing.assert_close(calibration["stats"][task][0]["std"], reference.std(dim=0, unbiased=False))
        ky = gaussian_kernel(batches[0][1], calibration["sigma_normals_squared"])
        perm = torch.randperm(8, generator=rng)
        first = normalized_information(torch.ones_like(ky), ky)
        second = normalized_information(torch.ones_like(ky), ky[perm][:, perm])
        self.assertAlmostEqual(first["sy"], second["sy"], places=12)
