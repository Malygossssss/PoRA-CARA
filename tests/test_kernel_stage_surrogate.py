import unittest

import numpy as np

from kernel_stage_search.surrogate import CANDIDATES, categorical_kernel, propose, select_winner


class SurrogateTests(unittest.TestCase):
    def test_categories_are_unordered(self):
        self.assertEqual(len(CANDIDATES), 81)
        first = ["BBBB", "BPGG", "PPPP"]
        rename = str.maketrans({"B": "G", "P": "B", "G": "P"})
        np.testing.assert_allclose(categorical_kernel(first, first), categorical_kernel([x.translate(rename) for x in first], [x.translate(rename) for x in first]))

    def test_twelve_distinct_actual_observations_and_resume(self):
        observations = []
        for index in range(12):
            proposal = propose(observations, 12)
            self.assertNotIn(proposal["modes"], [o["modes"] for o in observations])
            if index < 3:
                self.assertEqual(proposal["modes"], ["BBBB", "PPPP", "GGGG"][index])
            observations.append({"modes": proposal["modes"], "J": float(index % 4)})
        self.assertIsNone(propose(observations, 12))
        self.assertEqual(select_winner(observations), observations[3])
        self.assertEqual(propose(observations[:7], 12), propose(list(observations[:7]), 12))

    def test_constant_scores_have_finite_deterministic_posterior(self):
        observations = [{"modes": m, "J": 1.} for m in ["BBBB", "PPPP", "GGGG"]]
        p = propose(observations, 12)
        self.assertTrue(np.isfinite(p["acquisition"]))
        self.assertEqual(p["mu"], 0.)
        self.assertGreater(p["jitter"], 0)
        with self.assertRaises(ValueError):
            propose(observations + observations[:1], 12)
