import torch
import torch.distributed as dist


def should_restore_controller_state(restored_train_state):
    return bool(restored_train_state)


def maybe_load_controller_state(controller, state_dict, restored_train_state):
    if controller is None or not should_restore_controller_state(restored_train_state):
        return False
    if not state_dict:
        return False
    controller.load_state_dict(state_dict)
    return True


def validate_warmup_resume_state(controller, has_controller_state, restored_train_state, start_epoch):
    if controller is None or not controller.active:
        return
    if controller.ref_mode != "warmup_loss":
        return
    if not should_restore_controller_state(restored_train_state):
        return
    if has_controller_state:
        return
    if int(start_epoch) >= int(controller.warmup_epochs):
        raise RuntimeError(
            "Constrained MTL warmup_loss resume requires checkpoint extra_state.constrained_mtl "
            f"once TRAIN.START_EPOCH ({int(start_epoch)}) reaches TRAIN.CONSTRAINED_MTL.WARMUP_EPOCHS "
            f"({int(controller.warmup_epochs)}). Please resume from a new checkpoint that saved constrained state."
        )


class ConstrainedMTLController:
    DEFAULT_EPS_RELAX = 0.03
    REF_EPS = 1e-8
    RHO_GROWTH_THRESHOLD = 1e-4
    REF_TAIL_EPOCHS = 10

    def __init__(self, config, tasks, logger=None):
        cfg = config.TRAIN.CONSTRAINED_MTL

        self.logger = logger
        self.tasks = list(tasks)
        self.enabled = bool(getattr(config, "MTL", False) and cfg.ENABLED)

        protected_tasks = self._normalize_task_list(cfg.PROTECTED_TASKS)
        invalid_tasks = [task for task in protected_tasks if task not in self.tasks]
        if invalid_tasks:
            raise ValueError(
                f"TRAIN.CONSTRAINED_MTL.PROTECTED_TASKS contains unknown tasks: {invalid_tasks}"
            )
        self.protected_tasks = protected_tasks
        self.unconstrained_tasks = [
            task for task in self.tasks if task not in self.protected_tasks
        ]
        self.active = self.enabled and len(self.protected_tasks) > 0

        self.objective_mode = str(cfg.OBJECTIVE)
        if self.objective_mode not in {"avg_unconstrained", "avg_all"}:
            raise ValueError(
                f"Unsupported constrained objective '{self.objective_mode}'."
            )

        self.ref_mode = str(cfg.REF_MODE)
        if self.ref_mode not in {"baseline_loss", "warmup_loss"}:
            raise ValueError(f"Unsupported constrained ref mode '{self.ref_mode}'.")

        self.warmup_epochs = max(int(cfg.WARMUP_EPOCHS), 0)
        ref_epoch_start = int(cfg.REF_EPOCH_START)
        ref_epoch_end = int(cfg.REF_EPOCH_END)
        if self.ref_mode == "warmup_loss" and ref_epoch_end <= ref_epoch_start:
            if ref_epoch_start == 0 and ref_epoch_end == 0:
                ref_epoch_end = max(self.warmup_epochs - 1, 0)
            else:
                ref_epoch_end = ref_epoch_start
        if self.ref_mode == "warmup_loss":
            ref_epoch_start = max(
                ref_epoch_start,
                ref_epoch_end - self.REF_TAIL_EPOCHS + 1,
            )
        self.ref_epoch_start = ref_epoch_start
        self.ref_epoch_end = ref_epoch_end

        self.use_relative_loss = True
        if self.active and not bool(cfg.USE_RELATIVE_LOSS):
            self._warn_once(
                "_warned_relative_loss",
                "TRAIN.CONSTRAINED_MTL.USE_RELATIVE_LOSS=False is not supported; using relative normalization.",
            )

        self.dual_update_freq = str(cfg.DUAL_UPDATE_FREQ)
        if self.active and self.dual_update_freq != "epoch":
            raise ValueError("Only epoch-level dual updates are supported in constrained MTL v1.")

        self.dual_lr = float(cfg.DUAL_LR)
        self.dual_clamp_max = float(cfg.DUAL_CLAMP_MAX)
        self.rho = float(cfg.ALM_RHO)
        self.rho_growth = float(cfg.ALM_RHO_GROWTH)
        self.rho_patience = max(int(cfg.ALM_RHO_PATIENCE), 1)
        self.violation_ema_decay = float(cfg.VIOLATION_EMA)

        eps_relax_cfg = self._cfg_to_dict(cfg.EPS_RELAX)
        self.eps_relax = {
            task: float(eps_relax_cfg.get(task, self.DEFAULT_EPS_RELAX))
            for task in self.protected_tasks
        }

        self.lambdas = {task: 0.0 for task in self.protected_tasks}
        self.violation_ema = {task: 0.0 for task in self.protected_tasks}
        self.rho_patience_counter = {task: 0 for task in self.protected_tasks}

        self.ref_losses = {}
        self.ref_loss_frozen = False
        self.ref_loss_frozen_epoch = None
        self.ref_loss_sums = {task: 0.0 for task in self.tasks}
        self.ref_loss_counts = {task: 0.0 for task in self.tasks}

        self.epoch_violation_sums = {task: 0.0 for task in self.protected_tasks}
        self.epoch_pos_violation_sums = {task: 0.0 for task in self.protected_tasks}
        self.epoch_violation_counts = {task: 0.0 for task in self.protected_tasks}
        self.metric_device = None

        if self.active and self.ref_mode == "baseline_loss":
            ref_losses_cfg = self._cfg_to_dict(cfg.REF_LOSSES)
            missing_ref_tasks = [task for task in self.tasks if task not in ref_losses_cfg]
            if missing_ref_tasks:
                raise ValueError(
                    "TRAIN.CONSTRAINED_MTL.REF_LOSSES must provide all active tasks when REF_MODE=baseline_loss. "
                    f"Missing: {missing_ref_tasks}"
                )
            self.ref_losses = {
                task: float(ref_losses_cfg[task]) for task in self.tasks
            }
            self.ref_loss_frozen = True
            self.ref_loss_frozen_epoch = -1

    @staticmethod
    def _normalize_task_list(task_value):
        if task_value is None:
            return []
        if isinstance(task_value, str):
            values = [item.strip() for item in task_value.split(",")]
            return [item for item in values if item]
        ordered = []
        seen = set()
        for item in task_value:
            if item in seen:
                continue
            ordered.append(item)
            seen.add(item)
        return ordered

    @staticmethod
    def _cfg_to_dict(cfg_node):
        if cfg_node is None:
            return {}
        if hasattr(cfg_node, "items"):
            return {key: value for key, value in cfg_node.items()}
        return dict(cfg_node)

    def _warn_once(self, attr_name, message):
        if getattr(self, attr_name, False):
            return
        if self.logger is not None:
            self.logger.warning(message)
        setattr(self, attr_name, True)

    def _extract_task_losses(self, raw_loss_dict):
        task_losses = {}
        for task in self.tasks:
            if task not in raw_loss_dict:
                raise KeyError(f"Missing raw task loss for '{task}' in constrained controller.")
            task_losses[task] = raw_loss_dict[task].float()
        return task_losses

    def _should_collect_reference(self, epoch):
        if not self.active or self.ref_mode != "warmup_loss" or self.ref_loss_frozen:
            return False
        return self.ref_epoch_start <= epoch <= self.ref_epoch_end

    def _should_use_baseline(self, epoch):
        if not self.active:
            return True
        if self.ref_mode != "warmup_loss":
            return False
        if not self.ref_loss_frozen:
            return True
        return epoch < self.warmup_epochs

    def _phase_for_epoch(self, epoch):
        if not self.active:
            return "inactive"
        if self._should_use_baseline(epoch):
            return "warmup"
        if not self.ref_loss_frozen:
            return "waiting_ref"
        return "constrained"

    def _accumulate_reference_losses(self, task_losses, batch_weight):
        for task, loss_value in task_losses.items():
            self.ref_loss_sums[task] += float(loss_value.detach().item()) * batch_weight
            self.ref_loss_counts[task] += batch_weight

    def _dist_device(self):
        if self.metric_device is not None:
            return self.metric_device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _sync_reference_statistics(self):
        if not self.tasks:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return

        stats_values = []
        for task in self.tasks:
            stats_values.extend([
                self.ref_loss_sums[task],
                self.ref_loss_counts[task],
            ])
        stats_tensor = torch.tensor(
            stats_values,
            dtype=torch.float64,
            device=self._dist_device(),
        )
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        reduced_values = stats_tensor.tolist()
        for index, task in enumerate(self.tasks):
            base = index * 2
            self.ref_loss_sums[task] = reduced_values[base]
            self.ref_loss_counts[task] = reduced_values[base + 1]

    def _broadcast_ref_losses(self):
        if not self.ref_losses:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return

        ref_tensor = torch.tensor(
            [self.ref_losses[task] for task in self.tasks],
            dtype=torch.float64,
            device=self._dist_device(),
        )
        dist.broadcast(ref_tensor, src=0)
        for index, task in enumerate(self.tasks):
            self.ref_losses[task] = float(ref_tensor[index].item())

    def _freeze_reference_losses(self, epoch):
        if self.ref_loss_frozen:
            return False
        if self.ref_mode != "warmup_loss":
            return False
        if epoch < self.ref_epoch_end:
            return False

        self._sync_reference_statistics()
        missing_counts = [task for task in self.tasks if self.ref_loss_counts[task] <= 0.0]
        if missing_counts:
            self._warn_once(
                "_warned_missing_ref",
                "Constrained MTL reference losses are incomplete after warmup; continuing with baseline loss until ref_loss is available.",
            )
            return False

        self.ref_losses = {
            task: self.ref_loss_sums[task] / max(self.ref_loss_counts[task], 1.0)
            for task in self.tasks
        }
        self.ref_loss_frozen = True
        self.ref_loss_frozen_epoch = int(epoch)
        self._broadcast_ref_losses()
        return True

    def _compute_norm_losses(self, task_losses):
        return {
            task: loss_value / (float(self.ref_losses[task]) + self.REF_EPS)
            for task, loss_value in task_losses.items()
        }

    def _objective_tasks(self):
        if self.objective_mode == "avg_all":
            return list(self.tasks)
        if self.unconstrained_tasks:
            return list(self.unconstrained_tasks)
        return list(self.tasks)

    def _build_step_metrics(
        self,
        phase,
        task_losses,
        loss_to_optimize,
        objective_loss=None,
        norm_losses=None,
        violations=None,
    ):
        metrics = {
            "phase": phase,
            "objective_loss": None if objective_loss is None else float(objective_loss.detach().item()),
            "constrained_total_loss": float(loss_to_optimize.detach().item()),
            "alm_rho": float(self.rho),
        }

        if self.ref_loss_frozen:
            metrics["ref_loss"] = {
                task: float(self.ref_losses[task]) for task in self.tasks
            }
        else:
            metrics["ref_loss"] = {}

        metrics["raw_loss"] = {
            task: float(task_losses[task].detach().item()) for task in self.tasks
        }
        metrics["lambda"] = {
            task: float(self.lambdas[task]) for task in self.protected_tasks
        }

        if norm_losses is not None:
            metrics["norm_loss"] = {
                task: float(norm_losses[task].detach().item()) for task in self.tasks
            }
        else:
            metrics["norm_loss"] = {}

        if violations is not None:
            metrics["violation"] = {
                task: float(violations[task].detach().item()) for task in self.protected_tasks
            }
            metrics["pos_violation"] = {
                task: float(torch.relu(violations[task]).detach().item())
                for task in self.protected_tasks
            }
            metrics["constraint_satisfied"] = {
                task: float(violations[task].detach().item() <= 0.0)
                for task in self.protected_tasks
            }
        else:
            metrics["violation"] = {}
            metrics["pos_violation"] = {}
            metrics["constraint_satisfied"] = {}

        return metrics

    def compute_loss(self, raw_loss_dict, baseline_total_loss, epoch, batch_size):
        if not self.active:
            return baseline_total_loss, None

        task_losses = self._extract_task_losses(raw_loss_dict)
        batch_weight = float(batch_size)
        if self._should_collect_reference(epoch):
            self._accumulate_reference_losses(task_losses, batch_weight)

        phase = self._phase_for_epoch(epoch)
        if self._should_use_baseline(epoch) or not self.ref_loss_frozen:
            return baseline_total_loss, self._build_step_metrics(
                phase=phase,
                task_losses=task_losses,
                loss_to_optimize=baseline_total_loss,
            )

        norm_losses = self._compute_norm_losses(task_losses)
        objective_terms = [norm_losses[task] for task in self._objective_tasks()]
        objective_loss = torch.stack(objective_terms).mean()

        violations = {}
        lagrangian_term = objective_loss.new_zeros(())
        penalty_term = objective_loss.new_zeros(())
        for task in self.protected_tasks:
            violation = norm_losses[task] - (1.0 + self.eps_relax[task])
            violations[task] = violation
            lagrangian_term = lagrangian_term + violation * float(self.lambdas[task])
            positive_violation = torch.relu(violation)
            penalty_term = penalty_term + 0.5 * float(self.rho) * positive_violation.pow(2)

            violation_value = float(violation.detach().item())
            positive_value = float(torch.relu(violation).detach().item())
            self.epoch_violation_sums[task] += violation_value * batch_weight
            self.epoch_pos_violation_sums[task] += positive_value * batch_weight
            self.epoch_violation_counts[task] += batch_weight

        self.metric_device = objective_loss.device
        constrained_total_loss = objective_loss + lagrangian_term + penalty_term

        return constrained_total_loss, self._build_step_metrics(
            phase=phase,
            task_losses=task_losses,
            loss_to_optimize=constrained_total_loss,
            objective_loss=objective_loss,
            norm_losses=norm_losses,
            violations=violations,
        )

    def _reduce_epoch_stats(self):
        if not self.protected_tasks:
            return {}, {}

        avg_violation = {
            task: (
                self.epoch_violation_sums[task] / self.epoch_violation_counts[task]
                if self.epoch_violation_counts[task] > 0.0 else 0.0
            )
            for task in self.protected_tasks
        }
        avg_pos_violation = {
            task: (
                self.epoch_pos_violation_sums[task] / self.epoch_violation_counts[task]
                if self.epoch_violation_counts[task] > 0.0 else 0.0
            )
            for task in self.protected_tasks
        }

        if not (dist.is_available() and dist.is_initialized()):
            return avg_violation, avg_pos_violation

        tensor_values = []
        for task in self.protected_tasks:
            tensor_values.extend([
                self.epoch_violation_sums[task],
                self.epoch_pos_violation_sums[task],
                self.epoch_violation_counts[task],
            ])
        reduce_device = self.metric_device
        if reduce_device is None:
            reduce_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stats_tensor = torch.tensor(
            tensor_values,
            dtype=torch.float64,
            device=reduce_device,
        )
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        reduced = stats_tensor.tolist()

        avg_violation = {}
        avg_pos_violation = {}
        for index, task in enumerate(self.protected_tasks):
            base = index * 3
            count = reduced[base + 2]
            if count > 0.0:
                avg_violation[task] = reduced[base] / count
                avg_pos_violation[task] = reduced[base + 1] / count
            else:
                avg_violation[task] = 0.0
                avg_pos_violation[task] = 0.0
        return avg_violation, avg_pos_violation

    def _reset_epoch_stats(self):
        for task in self.protected_tasks:
            self.epoch_violation_sums[task] = 0.0
            self.epoch_pos_violation_sums[task] = 0.0
            self.epoch_violation_counts[task] = 0.0
        self.metric_device = None

    def on_epoch_end(self, epoch):
        if not self.active:
            return None

        ref_frozen_now = self._freeze_reference_losses(epoch)
        phase = self._phase_for_epoch(epoch)
        summary = {
            "phase": phase,
            "ref_loss": {
                task: float(self.ref_losses[task]) for task in self.tasks
            } if self.ref_loss_frozen else {},
            "lambda_after_update": {
                task: float(self.lambdas[task]) for task in self.protected_tasks
            },
            "alm_rho": float(self.rho),
            "ref_loss_frozen": bool(self.ref_loss_frozen),
            "ref_loss_frozen_epoch": self.ref_loss_frozen_epoch,
            "ref_frozen_this_epoch": ref_frozen_now,
        }

        if phase != "constrained":
            self._reset_epoch_stats()
            return summary

        avg_violation, avg_pos_violation = self._reduce_epoch_stats()
        lambda_before_update = {
            task: float(self.lambdas[task]) for task in self.protected_tasks
        }

        for task in self.protected_tasks:
            updated_lambda = max(
                0.0,
                self.lambdas[task] + self.dual_lr * avg_violation[task],
            )
            self.lambdas[task] = min(updated_lambda, self.dual_clamp_max)

            self.violation_ema[task] = (
                self.violation_ema_decay * self.violation_ema[task]
                + (1.0 - self.violation_ema_decay) * avg_pos_violation[task]
            )
            if self.violation_ema[task] > self.RHO_GROWTH_THRESHOLD:
                self.rho_patience_counter[task] += 1
            else:
                self.rho_patience_counter[task] = 0

        rho_grew = False
        if self.rho_growth > 1.0 and any(
            self.rho_patience_counter[task] >= self.rho_patience
            for task in self.protected_tasks
        ):
            self.rho *= self.rho_growth
            for task in self.protected_tasks:
                self.rho_patience_counter[task] = 0
            rho_grew = True

        summary.update({
            "epoch_avg_violation": avg_violation,
            "epoch_avg_pos_violation": avg_pos_violation,
            "lambda_before_update": lambda_before_update,
            "lambda_after_update": {
                task: float(self.lambdas[task]) for task in self.protected_tasks
            },
            "violation_ema": {
                task: float(self.violation_ema[task]) for task in self.protected_tasks
            },
            "alm_rho": float(self.rho),
            "alm_rho_grew": rho_grew,
        })

        self._reset_epoch_stats()
        return summary

    def format_step_summary(self, step_metrics):
        if not step_metrics:
            return None

        summary = [
            f"phase={step_metrics['phase']}",
            f"rho={step_metrics['alm_rho']:.4f}",
        ]
        if step_metrics["objective_loss"] is not None:
            summary.append(f"objective={step_metrics['objective_loss']:.4f}")
        if self.protected_tasks:
            summary.extend(
                f"{task}:lambda={step_metrics['lambda'].get(task, 0.0):.4f}"
                for task in self.protected_tasks
            )
            if step_metrics["norm_loss"]:
                summary.extend(
                    f"{task}:norm={step_metrics['norm_loss'][task]:.4f}"
                    for task in self.protected_tasks
                )
            if step_metrics["violation"]:
                summary.extend(
                    f"{task}:g={step_metrics['violation'][task]:.4f}"
                    for task in self.protected_tasks
                )
        return " | ".join(summary)

    def format_epoch_summary(self, epoch_summary):
        if not epoch_summary:
            return None

        chunks = [
            f"phase={epoch_summary['phase']}",
            f"rho={epoch_summary['alm_rho']:.4f}",
        ]
        if epoch_summary.get("ref_frozen_this_epoch"):
            chunks.append("ref_loss=frozen")
        if epoch_summary.get("ref_loss"):
            chunks.extend(
                f"ref[{task}]={epoch_summary['ref_loss'][task]:.4f}"
                for task in self.tasks
            )
        if "epoch_avg_violation" in epoch_summary:
            chunks.extend(
                f"{task}:avg_g={epoch_summary['epoch_avg_violation'][task]:.4f}"
                for task in self.protected_tasks
            )
            chunks.extend(
                f"{task}:avg_pos_g={epoch_summary['epoch_avg_pos_violation'][task]:.4f}"
                for task in self.protected_tasks
            )
            chunks.extend(
                f"{task}:lambda={epoch_summary['lambda_after_update'][task]:.4f}"
                for task in self.protected_tasks
            )
        return " | ".join(chunks)

    def state_dict(self):
        return {
            "tasks": list(self.tasks),
            "protected_tasks": list(self.protected_tasks),
            "ref_losses": dict(self.ref_losses),
            "ref_loss_frozen": bool(self.ref_loss_frozen),
            "ref_loss_frozen_epoch": self.ref_loss_frozen_epoch,
            "ref_loss_sums": dict(self.ref_loss_sums),
            "ref_loss_counts": dict(self.ref_loss_counts),
            "lambdas": dict(self.lambdas),
            "rho": float(self.rho),
            "violation_ema": dict(self.violation_ema),
            "rho_patience_counter": dict(self.rho_patience_counter),
        }

    def load_state_dict(self, state_dict):
        if not state_dict or not self.active:
            return

        self.ref_losses = {
            task: float(value)
            for task, value in state_dict.get("ref_losses", {}).items()
            if task in self.tasks
        }
        self.ref_loss_frozen = bool(state_dict.get("ref_loss_frozen", False))
        self.ref_loss_frozen_epoch = state_dict.get("ref_loss_frozen_epoch")

        for task in self.tasks:
            if task in state_dict.get("ref_loss_sums", {}):
                self.ref_loss_sums[task] = float(state_dict["ref_loss_sums"][task])
            if task in state_dict.get("ref_loss_counts", {}):
                self.ref_loss_counts[task] = float(state_dict["ref_loss_counts"][task])

        for task in self.protected_tasks:
            if task in state_dict.get("lambdas", {}):
                self.lambdas[task] = float(state_dict["lambdas"][task])
            if task in state_dict.get("violation_ema", {}):
                self.violation_ema[task] = float(state_dict["violation_ema"][task])
            if task in state_dict.get("rho_patience_counter", {}):
                self.rho_patience_counter[task] = int(state_dict["rho_patience_counter"][task])

        if "rho" in state_dict:
            self.rho = float(state_dict["rho"])
