import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from kernel_stage_search.features import sample_stage, collect_batch_features
from kernel_stage_search.protocol import TASKS
from test_kernel_stage_data import TinyDataset


class FeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        for layer, size in zip(self.layers, [2, 1, 1, 1]):
            layer.output_resolution = (size, size)

    def forward(self, images, task=None, return_stages=False):
        return [F.adaptive_avg_pool2d(images[:, :1], layer.output_resolution).flatten(2).transpose(1, 2) + list(TASKS).index(task)
                for layer in self.layers]


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FeatureBackbone()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, images):
        if not torch.is_inference_mode_enabled() or torch.is_grad_enabled():
            raise AssertionError("Feature scoring must use frozen inference mode")
        return {t: self.backbone(images, task=t, return_stages=True)[-1] for t in TASKS}


class FeatureTests(unittest.TestCase):
    def test_real_task_hook_matches_direct_backbone_features(self):
        from kernel_stage_search.synthetic import tiny_config, SyntheticDataset
        from kernel_stage_search.data import build_sample_plan, index_by_id
        from models import build_model, build_mtl_model
        config = tiny_config()
        model = build_mtl_model(build_model(config), config).eval().requires_grad_(False)
        dataset = SyntheticDataset()
        batch = build_sample_plan(dataset, dataset.im_ids, 1, 2, 4, 8)["batches"][0]
        actual = collect_batch_features(model, dataset, batch, microbatch=2)
        indices = index_by_id(dataset)
        images = torch.stack([dataset[indices[name]]["image"] for name in batch["image_ids"]])
        with torch.inference_mode():
            for task in TASKS:
                direct = model.backbone(images, task=task, return_stages=True)
                for stage, tokens in enumerate(direct):
                    values, rows = sample_stage(tokens, model.backbone.layers[stage].output_resolution,
                                                batch["image_ids"], batch["tasks"][task]["coordinates"], batch["label_shapes"])
                    expected = torch.empty_like(actual[task][stage])
                    expected[rows] = values
                    torch.testing.assert_close(actual[task][stage], expected)

    def test_pixel_center_matches_interpolate_then_gather(self):
        feature = torch.arange(12.).reshape(1, 12, 1)
        coordinates = [["a", y, x] for y in range(7) for x in range(9)]
        sampled, rows = sample_stage(feature, (3, 4), ["a"], coordinates, {"a": [7, 9]})
        reference = F.interpolate(feature.transpose(1, 2).reshape(1, 1, 3, 4), size=(7, 9), mode="bilinear", align_corners=False)
        torch.testing.assert_close(sampled[:, 0], reference.flatten(), atol=1e-6, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "token"):
            sample_stage(torch.zeros(1, 14, 1), (3, 4), ["a"], coordinates, {"a": [7, 9]})

    def test_task_hook_and_image_order_are_exact(self):
        from kernel_stage_search.data import build_sample_plan
        dataset = TinyDataset(True)
        plan = build_sample_plan(dataset, dataset.im_ids, 1, 2, 4, 8)["batches"][0]
        model = FeatureModel().eval()
        forward = collect_batch_features(model, dataset, plan, microbatch=1)
        plan["image_ids"].reverse()
        reverse = collect_batch_features(model, dataset, plan, microbatch=2)
        for task in TASKS:
            for left, right in zip(forward[task], reverse[task]):
                torch.testing.assert_close(left, right)
        self.assertFalse(model.backbone._forward_hooks)
