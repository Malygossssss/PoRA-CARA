import unittest
from unittest import mock

import torch
from yacs.config import CfgNode as CN

from constrained_mtl import (
    ConstrainedMTLController,
    maybe_load_controller_state,
    validate_warmup_resume_state,
)


def build_config(
    enabled=True,
    protected_tasks=None,
    ref_mode="warmup_loss",
    warmup_epochs=1,
    objective="avg_unconstrained",
    ref_losses=None,
    eps_relax=None,
):
    cfg = CN()
    cfg.MTL = True
    cfg.TRAIN = CN()
    cfg.TRAIN.CONSTRAINED_MTL = CN()
    cfg.TRAIN.CONSTRAINED_MTL.ENABLED = enabled
    cfg.TRAIN.CONSTRAINED_MTL.PROTECTED_TASKS = protected_tasks or []
    cfg.TRAIN.CONSTRAINED_MTL.OBJECTIVE = objective
    cfg.TRAIN.CONSTRAINED_MTL.REF_MODE = ref_mode
    cfg.TRAIN.CONSTRAINED_MTL.REF_EPOCH_START = 0
    cfg.TRAIN.CONSTRAINED_MTL.REF_EPOCH_END = 0
    cfg.TRAIN.CONSTRAINED_MTL.WARMUP_EPOCHS = warmup_epochs
    cfg.TRAIN.CONSTRAINED_MTL.USE_RELATIVE_LOSS = True
    cfg.TRAIN.CONSTRAINED_MTL.EPS_RELAX = CN(new_allowed=True)
    cfg.TRAIN.CONSTRAINED_MTL.REF_LOSSES = CN(new_allowed=True)
    cfg.TRAIN.CONSTRAINED_MTL.DUAL_UPDATE_FREQ = "epoch"
    cfg.TRAIN.CONSTRAINED_MTL.DUAL_LR = 0.05
    cfg.TRAIN.CONSTRAINED_MTL.DUAL_CLAMP_MAX = 10.0
    cfg.TRAIN.CONSTRAINED_MTL.ALM_RHO = 1.0
    cfg.TRAIN.CONSTRAINED_MTL.ALM_RHO_GROWTH = 1.5
    cfg.TRAIN.CONSTRAINED_MTL.ALM_RHO_PATIENCE = 5
    cfg.TRAIN.CONSTRAINED_MTL.VIOLATION_EMA = 0.9

    for task, value in (eps_relax or {}).items():
        cfg.TRAIN.CONSTRAINED_MTL.EPS_RELAX[task] = value
    for task, value in (ref_losses or {}).items():
        cfg.TRAIN.CONSTRAINED_MTL.REF_LOSSES[task] = value
    return cfg


def build_loss_dict(main_loss, sal_loss):
    return {
        "main": torch.tensor(float(main_loss), requires_grad=True),
        "sal": torch.tensor(float(sal_loss), requires_grad=True),
        "total": torch.tensor(float(main_loss + sal_loss), requires_grad=True),
    }


class ConstrainedMTLControllerTest(unittest.TestCase):
    def test_empty_protected_tasks_falls_back_to_baseline(self):
        config = build_config(protected_tasks=[])
        controller = ConstrainedMTLController(config, ["main", "sal"])

        baseline_total = torch.tensor(6.0, requires_grad=True)
        loss, metrics = controller.compute_loss(
            build_loss_dict(2.0, 4.0), baseline_total, epoch=0, batch_size=2
        )

        self.assertFalse(controller.active)
        self.assertIs(loss, baseline_total)
        self.assertIsNone(metrics)

    def test_warmup_freezes_ref_loss_and_updates_lambda_at_epoch_end(self):
        config = build_config(
            protected_tasks=["sal"],
            ref_mode="warmup_loss",
            warmup_epochs=1,
            eps_relax={"sal": 0.03},
        )
        controller = ConstrainedMTLController(config, ["main", "sal"])

        baseline_total = torch.tensor(6.0, requires_grad=True)
        warmup_loss, warmup_metrics = controller.compute_loss(
            build_loss_dict(2.0, 4.0), baseline_total, epoch=0, batch_size=2
        )
        warmup_summary = controller.on_epoch_end(0)

        self.assertAlmostEqual(warmup_loss.item(), 6.0, places=6)
        self.assertEqual(warmup_metrics["phase"], "warmup")
        self.assertTrue(warmup_summary["ref_loss_frozen"])
        self.assertAlmostEqual(warmup_summary["ref_loss"]["main"], 2.0, places=6)
        self.assertAlmostEqual(warmup_summary["ref_loss"]["sal"], 4.0, places=6)

        constrained_total, constrained_metrics = controller.compute_loss(
            build_loss_dict(2.0, 4.4),
            torch.tensor(6.4, requires_grad=True),
            epoch=1,
            batch_size=2,
        )

        self.assertEqual(constrained_metrics["phase"], "constrained")
        self.assertAlmostEqual(constrained_metrics["norm_loss"]["main"], 1.0, places=6)
        self.assertAlmostEqual(constrained_metrics["norm_loss"]["sal"], 1.1, places=6)
        self.assertAlmostEqual(constrained_metrics["violation"]["sal"], 0.07, places=6)
        self.assertAlmostEqual(constrained_total.item(), 1.00245, places=6)

        epoch_summary = controller.on_epoch_end(1)
        self.assertAlmostEqual(epoch_summary["epoch_avg_violation"]["sal"], 0.07, places=6)
        self.assertAlmostEqual(epoch_summary["lambda_after_update"]["sal"], 0.0035, places=6)

    def test_warmup_reference_uses_last_ten_epochs(self):
        config = build_config(
            protected_tasks=["sal"],
            ref_mode="warmup_loss",
            warmup_epochs=12,
        )
        controller = ConstrainedMTLController(config, ["main", "sal"])

        self.assertEqual(controller.ref_epoch_start, 2)
        self.assertEqual(controller.ref_epoch_end, 11)

        for epoch in range(12):
            main_loss = float(epoch + 1)
            sal_loss = float((epoch + 1) * 2)
            baseline_total = torch.tensor(main_loss + sal_loss, requires_grad=True)
            controller.compute_loss(
                build_loss_dict(main_loss, sal_loss),
                baseline_total,
                epoch=epoch,
                batch_size=1,
            )

        warmup_summary = controller.on_epoch_end(11)

        self.assertTrue(warmup_summary["ref_loss_frozen"])
        self.assertAlmostEqual(warmup_summary["ref_loss"]["main"], 7.5, places=6)
        self.assertAlmostEqual(warmup_summary["ref_loss"]["sal"], 15.0, places=6)

    def test_objective_falls_back_to_avg_all_when_no_unconstrained_tasks(self):
        config = build_config(
            protected_tasks=["main", "sal"],
            ref_mode="baseline_loss",
            warmup_epochs=0,
            objective="avg_unconstrained",
            ref_losses={"main": 2.0, "sal": 4.0},
            eps_relax={"main": 0.03, "sal": 0.03},
        )
        controller = ConstrainedMTLController(config, ["main", "sal"])

        constrained_total, metrics = controller.compute_loss(
            build_loss_dict(2.2, 4.4),
            torch.tensor(6.6, requires_grad=True),
            epoch=0,
            batch_size=2,
        )

        self.assertEqual(metrics["phase"], "constrained")
        self.assertAlmostEqual(metrics["objective_loss"], 1.1, places=6)
        self.assertAlmostEqual(constrained_total.item(), 1.1049, places=6)

    def test_state_dict_round_trip_restores_dual_state(self):
        config = build_config(
            protected_tasks=["sal"],
            ref_mode="baseline_loss",
            warmup_epochs=0,
            ref_losses={"main": 2.0, "sal": 4.0},
            eps_relax={"sal": 0.03},
        )
        controller = ConstrainedMTLController(config, ["main", "sal"])

        controller.compute_loss(
            build_loss_dict(2.0, 4.4),
            torch.tensor(6.4, requires_grad=True),
            epoch=0,
            batch_size=2,
        )
        controller.on_epoch_end(0)
        state = controller.state_dict()

        restored = ConstrainedMTLController(config, ["main", "sal"])
        restored.load_state_dict(state)

        self.assertTrue(restored.ref_loss_frozen)
        self.assertAlmostEqual(restored.ref_losses["main"], 2.0, places=6)
        self.assertAlmostEqual(restored.ref_losses["sal"], 4.0, places=6)
        self.assertAlmostEqual(
            restored.lambdas["sal"], controller.lambdas["sal"], places=6
        )
        self.assertAlmostEqual(restored.rho, controller.rho, places=6)

    def test_old_checkpoint_without_controller_state_raises_after_warmup(self):
        config = build_config(
            protected_tasks=["sal"],
            ref_mode="warmup_loss",
            warmup_epochs=2,
        )
        controller = ConstrainedMTLController(config, ["main", "sal"])

        with self.assertRaisesRegex(RuntimeError, "extra_state.constrained_mtl"):
            validate_warmup_resume_state(
                controller,
                has_controller_state=False,
                restored_train_state=True,
                start_epoch=2,
            )

    def test_partial_resume_does_not_load_controller_state(self):
        config = build_config(
            protected_tasks=["sal"],
            ref_mode="baseline_loss",
            warmup_epochs=0,
            ref_losses={"main": 2.0, "sal": 4.0},
        )
        controller = ConstrainedMTLController(config, ["main", "sal"])
        state = controller.state_dict()
        state["lambdas"]["sal"] = 1.25

        loaded = maybe_load_controller_state(
            controller,
            state,
            restored_train_state=False,
        )

        self.assertFalse(loaded)
        self.assertAlmostEqual(controller.lambdas["sal"], 0.0, places=6)

    def test_freeze_reference_losses_uses_global_ddp_stats(self):
        config = build_config(
            protected_tasks=["sal"],
            ref_mode="warmup_loss",
            warmup_epochs=1,
        )
        controller = ConstrainedMTLController(config, ["main", "sal"])
        controller.ref_loss_sums["main"] = 1.0
        controller.ref_loss_counts["main"] = 1.0
        controller.ref_loss_sums["sal"] = 4.0
        controller.ref_loss_counts["sal"] = 2.0

        all_reduce_calls = []
        broadcast_calls = []

        def fake_all_reduce(tensor, op=None):
            all_reduce_calls.append(tensor.clone())
            tensor.copy_(torch.tensor([3.0, 2.0, 7.5, 3.0], dtype=tensor.dtype, device=tensor.device))

        def fake_broadcast(tensor, src=0):
            broadcast_calls.append((tensor.clone(), src))

        with mock.patch("constrained_mtl.dist.is_available", return_value=True), \
             mock.patch("constrained_mtl.dist.is_initialized", return_value=True), \
             mock.patch("constrained_mtl.dist.all_reduce", side_effect=fake_all_reduce), \
             mock.patch("constrained_mtl.dist.broadcast", side_effect=fake_broadcast), \
             mock.patch("constrained_mtl.torch.cuda.is_available", return_value=False):
            frozen = controller._freeze_reference_losses(0)

        self.assertTrue(frozen)
        self.assertEqual(len(all_reduce_calls), 1)
        self.assertEqual(len(broadcast_calls), 1)
        self.assertAlmostEqual(controller.ref_loss_sums["main"], 3.0, places=6)
        self.assertAlmostEqual(controller.ref_loss_counts["main"], 2.0, places=6)
        self.assertAlmostEqual(controller.ref_loss_sums["sal"], 7.5, places=6)
        self.assertAlmostEqual(controller.ref_loss_counts["sal"], 3.0, places=6)
        self.assertAlmostEqual(controller.ref_losses["main"], 1.5, places=6)
        self.assertAlmostEqual(controller.ref_losses["sal"], 2.5, places=6)


if __name__ == "__main__":
    unittest.main()
