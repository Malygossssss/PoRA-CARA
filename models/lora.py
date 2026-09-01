# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Microsoft LoRA (https://github.com/microsoft/LoRA)
#
# Original file:
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
#
# Adapted file:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        tasks=None,
        **kwargs,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(
            in_features, out_features, **kwargs)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            # Merge the weights and mark it
            self.linear.weight.data += (self.lora_B @
                                        self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)) * self.scaling
        return pretrained + lora


class MTLoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        tasks=None,
        task_to_group: Mapping[str, str] = None,
        trainable_scale_shared=False,
        trainable_scale_per_task=False,
        shared_mode: str = 'matrix',
        layer_idx: int = -1,
        **kwargs,
    ):
        assert shared_mode in ['matrix', 'matrixv2',
                               'add', 'addition', 'lora_only']
        if shared_mode == 'add':
            shared_mode = 'addition'
        if shared_mode == 'lora_only':
            tasks = None
        has_tasks = tasks is not None
        if not has_tasks:
            if shared_mode not in ['matrix']:
                shared_mode = 'matrix'

        if isinstance(r, int):
            r = {'shared': r}
        r = dict(r)
        task_set = set(tasks or [])
        group_keys = [key for key in r.keys() if key not in task_set and key != 'shared']
        base_rank = int(r.get('shared', 0))
        if len(group_keys) > 0:
            base_rank = max(base_rank, max(int(r[key]) for key in group_keys))
        super().__init__(
            r=base_rank, lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)

        self.linear = torch.nn.Linear(
            in_features, out_features, **kwargs)
        self.layer_idx = layer_idx
        self.shared_lora_enabled = True

        self.tasks = tasks
        self.task_to_group = dict(task_to_group or {})
        self.shared_mode = shared_mode
        self.r_shared = int(r.get('shared', 0))
        self.group_shared_ranks = {group: int(r[group]) for group in group_keys}
        self.group_names = list(self.group_shared_ranks.keys())
        self.ag_enabled = len(self.group_names) > 0
        if self.ag_enabled and self.shared_mode == 'addition':
            raise NotImplementedError("AG-MTLoRA does not support shared_mode='addition'.")
        self.r_tasks = {task: int(r.get(task, 0)) for task in (tasks or [])}
        self.task_lora_enabled = {task: True for task in self.r_tasks.keys()}
        self.has_shared_lora = (self.r_shared > 0) or any(rank > 0 for rank in self.group_shared_ranks.values())
        self.has_task_lora = has_tasks and any(rank > 0 for rank in self.r_tasks.values())
        self.has_lora = self.has_shared_lora or self.has_task_lora

        if self.has_task_lora:
            self.lora_tasks_A = nn.ParameterDict({
                task: nn.Parameter(
                    self.linear.weight.new_zeros((self.r_tasks[task], in_features)))
                for task in tasks if self.r_tasks[task] > 0
            })
            self.lora_tasks_B = nn.ParameterDict({
                task: nn.Parameter(
                    self.linear.weight.new_zeros((out_features, self.r_tasks[task])))
                for task in tasks if self.r_tasks[task] > 0
            })
            if trainable_scale_per_task:
                self.lora_task_scale = nn.ParameterDict({
                    task: nn.Parameter(torch.FloatTensor(
                        [lora_task_scale[task] if isinstance(lora_task_scale, Mapping) else lora_task_scale]))
                    for task in self.lora_tasks_A.keys()
                })
            else:
                self.lora_task_scale = {
                    task: lora_task_scale[task] if isinstance(lora_task_scale, Mapping) else lora_task_scale
                    for task in self.lora_tasks_A.keys()
                }

        if self.has_shared_lora:
            if self.shared_mode == 'addition':
                assert has_tasks
                self.lora_norm = nn.LayerNorm(out_features)
            elif self.shared_mode == 'matrix' or self.shared_mode == 'matrixv2':
                if self.ag_enabled:
                    self.lora_shared_A_groups = nn.ParameterDict({
                        group: nn.Parameter(
                            self.linear.weight.new_zeros((self.group_shared_ranks[group], in_features))
                        )
                        for group in self.group_names if self.group_shared_ranks[group] > 0
                    })
                    self.lora_shared_B_groups = nn.ParameterDict({
                        group: nn.Parameter(
                            self.linear.weight.new_zeros((out_features, self.group_shared_ranks[group]))
                        )
                        for group in self.group_names if self.group_shared_ranks[group] > 0
                    })
                else:
                    self.lora_shared_A = nn.Parameter(
                        self.linear.weight.new_zeros((self.r_shared, in_features)))
                    self.lora_shared_B = nn.Parameter(
                        self.linear.weight.new_zeros((out_features, self.r_shared)))
            else:
                raise NotImplementedError
            if trainable_scale_shared and self.ag_enabled:
                self.lora_shared_scale = nn.ParameterDict({
                    group: nn.Parameter(torch.FloatTensor([lora_shared_scale]))
                    for group in self.group_names if self.group_shared_ranks[group] > 0
                })
            elif trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(
                    torch.FloatTensor([lora_shared_scale]))
            else:
                if self.ag_enabled:
                    self.lora_shared_scale = {
                        group: lora_shared_scale for group in self.group_names
                    }
                else:
                    self.lora_shared_scale = lora_shared_scale

        if self.has_shared_lora or self.has_task_lora:
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_shared_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)
        if hasattr(self, "lora_shared_A_groups"):
            for group in self.lora_shared_A_groups.keys():
                nn.init.kaiming_uniform_(self.lora_shared_A_groups[group], a=math.sqrt(5))
                nn.init.zeros_(self.lora_shared_B_groups[group])
        if hasattr(self, "lora_tasks_A"):
            for task in self.lora_tasks_A.keys():
                nn.init.kaiming_uniform_(
                    self.lora_tasks_A[task], a=math.sqrt(5))
                nn.init.zeros_(self.lora_tasks_B[task])

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        raise NotImplementedError

    def set_shared_lora_enabled(self, enabled: bool) -> None:
        self.shared_lora_enabled = bool(enabled)

    def set_task_lora_enabled(self, task: str, enabled: bool) -> None:
        if task in self.task_lora_enabled:
            self.task_lora_enabled[task] = bool(enabled)

    def get_task_lora_enabled(self, task: str) -> bool:
        return bool(self.task_lora_enabled.get(task, True))

    def _apply_task_lora(self, task: str, task_input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "lora_tasks_A") or task not in self.lora_tasks_A:
            return 0.0
        return (
            task_input
            @ self.lora_tasks_A[task].transpose(0, 1)
            @ self.lora_tasks_B[task].transpose(0, 1)
            * self.lora_task_scale[task]
        )

    def _get_group_scale(self, group_name: str):
        if isinstance(self.lora_shared_scale, nn.ParameterDict):
            return self.lora_shared_scale[group_name]
        if isinstance(self.lora_shared_scale, Mapping):
            return self.lora_shared_scale[group_name]
        return self.lora_shared_scale

    def _apply_group_shared_lora(self, group_name: str, task_input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "lora_shared_A_groups"):
            return 0.0
        if group_name not in self.lora_shared_A_groups:
            return 0.0
        return (
            task_input
            @ self.lora_shared_A_groups[group_name].transpose(0, 1)
            @ self.lora_shared_B_groups[group_name].transpose(0, 1)
        ) * self._get_group_scale(group_name)

    def forward(
        self,
        x: torch.Tensor,
        x_tasks: Dict[str, torch.Tensor] = None,
        active_task: str = None,
    ):
        # TODO: handle merging
        pretrained = self.linear(x)
        if not self.has_lora:
            return pretrained, None

        x = self.lora_dropout(x)
        if self.ag_enabled:
            if active_task is not None:
                if self.tasks is None or active_task not in self.tasks:
                    raise ValueError(
                        f"active_task must be one of {list(self.tasks or [])}, got {active_task!r}."
                    )
                if active_task not in self.task_to_group:
                    raise ValueError(
                        f"active_task {active_task!r} does not have an AG-MTLoRA group mapping."
                    )
                routed_tasks = (active_task,)
            else:
                routed_tasks = self.tasks or []
            lora_tasks = {}
            for task in routed_tasks:
                task_input_raw = x if x_tasks is None else x_tasks[task]
                task_input = self.lora_dropout(task_input_raw)
                task_pretrained = pretrained if x_tasks is None else self.linear(task_input_raw)
                task_output = task_pretrained

                if self.has_shared_lora and self.shared_lora_enabled:
                    task_output = task_output + self._apply_group_shared_lora(
                        self.task_to_group[task], task_input
                    )

                if self.has_task_lora and task in self.lora_tasks_A:
                    if self.get_task_lora_enabled(task):
                        task_output = task_output + self._apply_task_lora(task, task_input)

                lora_tasks[task] = task_output

            return pretrained, lora_tasks

        lora = 0.0
        lora_tasks = None

        if self.has_task_lora:
            lora_tasks = {}
            for task in self.lora_tasks_A.keys():
                if not self.get_task_lora_enabled(task):
                    lora_tasks[task] = pretrained
                    continue
                task_input = x if x_tasks is None else x_tasks[task]
                lora_tasks[task] = pretrained + self._apply_task_lora(task, task_input)

        if self.has_shared_lora and self.shared_lora_enabled:
            if self.shared_mode == 'matrix':
                lora = (x @ self.lora_shared_A.transpose(0, 1)
                        @ self.lora_shared_B.transpose(0, 1)) * self.lora_shared_scale
            elif self.shared_mode == 'matrixv2':
                lora = (x @ self.lora_shared_A.transpose(0, 1)
                        @ self.lora_shared_B.transpose(0, 1)) * self.lora_shared_scale
                if lora_tasks is not None:
                    lora_tasks = {task: task_out + lora for task, task_out in lora_tasks.items()}
            elif self.shared_mode == 'addition':
                if lora_tasks is None or len(lora_tasks) == 0:
                    lora = 0.0
                else:
                    lora = self.lora_norm(torch.sum(torch.stack(
                        list(lora_tasks.values()), dim=0), dim=0))
            else:
                raise NotImplementedError

        return pretrained + lora, lora_tasks


class MTLoRAQKV(LoRALayer):
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        tasks=None,
        task_to_group: Mapping[str, str] = None,
        trainable_scale_shared=False,
        trainable_scale_per_task=False,
        shared_mode: str = 'matrix',
        layer_idx: int = -1,
        **kwargs,
    ):
        if isinstance(r, int):
            r = {'shared': r}
        super().__init__(r=r, lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)
        self.tasks = tasks
        self.layer_idx = layer_idx
        self.q = MTLoRALinear(in_features, out_features, r=r, lora_shared_scale=lora_shared_scale, lora_task_scale=lora_task_scale, lora_dropout=lora_dropout,
                              tasks=tasks, task_to_group=task_to_group, trainable_scale_shared=trainable_scale_shared, trainable_scale_per_task=trainable_scale_per_task, shared_mode=shared_mode, layer_idx=layer_idx, **kwargs)
        self.k = MTLoRALinear(in_features, out_features, r=r, lora_shared_scale=lora_shared_scale, lora_task_scale=lora_task_scale, lora_dropout=lora_dropout,
                              tasks=tasks, task_to_group=task_to_group, trainable_scale_shared=trainable_scale_shared, trainable_scale_per_task=trainable_scale_per_task, shared_mode=shared_mode, layer_idx=layer_idx, **kwargs)
        self.v = MTLoRALinear(in_features, out_features, r=r, lora_shared_scale=lora_shared_scale, lora_task_scale=lora_task_scale, lora_dropout=lora_dropout,
                              tasks=tasks, task_to_group=task_to_group, trainable_scale_shared=trainable_scale_shared, trainable_scale_per_task=trainable_scale_per_task, shared_mode=shared_mode, layer_idx=layer_idx, **kwargs)
        self.has_task_lora = any(getattr(module, "has_task_lora", False) for module in (self.q, self.k, self.v))
        self.has_shared_lora = any(getattr(module, "has_shared_lora", False) for module in (self.q, self.k, self.v))
        self.shared_lora_enabled = True

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def merge(self):
        raise NotImplementedError

    def set_shared_lora_enabled(self, enabled: bool) -> None:
        self.shared_lora_enabled = bool(enabled)
        self.q.set_shared_lora_enabled(enabled)
        self.k.set_shared_lora_enabled(enabled)
        self.v.set_shared_lora_enabled(enabled)

    def set_task_lora_enabled(self, task: str, enabled: bool) -> None:
        self.q.set_task_lora_enabled(task, enabled)
        self.k.set_task_lora_enabled(task, enabled)
        self.v.set_task_lora_enabled(task, enabled)

    def get_task_lora_enabled(self, task: str) -> bool:
        return (
            self.q.get_task_lora_enabled(task)
            and self.k.get_task_lora_enabled(task)
            and self.v.get_task_lora_enabled(task)
        )

    def forward(
        self,
        x: torch.Tensor,
        x_tasks: Dict[str, torch.Tensor] = None,
        active_task: str = None,
    ):
        q_shared, q_tasks = self.q(x, x_tasks, active_task=active_task)
        k_shared, k_tasks = self.k(x, x_tasks, active_task=active_task)
        v_shared, v_tasks = self.v(x, x_tasks, active_task=active_task)

        task_outputs = None
        if self.tasks is not None and q_tasks is not None and k_tasks is not None and v_tasks is not None:
            task_outputs = {
                task: torch.cat([q_tasks[task], k_tasks[task], v_tasks[task]], dim=-1)
                for task in q_tasks.keys()
                if task in k_tasks and task in v_tasks
            }

        return torch.cat([q_shared, k_shared, v_shared], dim=-1), task_outputs


class LoRAQKVLinear(LoRALinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `lit_gpt/config.py`)
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA we can set it as False. For example if we want to apply LoRA only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
        """
        super(LoRALinear, self).__init__(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = [enable_lora] * 3
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.linear.weight.new_zeros(
                (r * sum(enable_lora), in_features)))  # (4, 128)
            enable_q, enable_k, enable_v = enable_lora
            self.kv_embd_size = self.linear.in_features // (
                n_head // n_query_groups)
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                self.linear.in_features * enable_q,
                self.kv_embd_size * enable_k,
                self.kv_embd_size * enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            self.lora_B = nn.Parameter(self.linear.weight.new_zeros(
                sum(self.qkv_shapes), r))  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Compute the indices
            # Indices are needed to properly pad weight updates with zeros. If we want to fine-tune queries and values,
            # but not keys, then the weights update should be:
            #
            # [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
            #  [....................................],
            #  [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            #      ↑              ↑            ↑
            # ________________________________________
            # | query         | key       | value    |
            # ----------------------------------------
            self.lora_ind = []
            if enable_q:
                self.lora_ind.extend(range(0, self.linear.in_features))
            if enable_k:
                self.lora_ind.extend(
                    range(self.linear.in_features, self.linear.in_features + self.kv_embd_size))
            if enable_v:
                self.lora_ind.extend(
                    range(self.linear.in_features + self.kv_embd_size, self.linear.out_features))
            self.reset_parameters()

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_lora):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)
        # Note: double transpose (in the beginning and in the end) is basically a guard for two-dimensional tensors
        # for example when we want to merge/unmerge LoRA weights and pretrained weights
        x = x.transpose(0, 1)
        result = x.new_zeros(
            (*x.shape[:-1], self.linear.out_features))  # (64, 64, 384)
        result = result.view(-1, self.linear.out_features)  # (4096, 384)
        result = result.index_copy(
            1, torch.tensor(
                self.lora_ind, device=result.device), x.reshape(-1, sum(self.qkv_shapes))
        )  # (4096, 256)
        # (64, 64, 384)
        return result.view((*x.shape[:-1], self.linear.out_features)).transpose(0, 1)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `lit_gpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be splitted in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """
        if self.n_head == self.n_query_groups:
            # (B, C_output, T)
            return F.conv1d(input, weight, groups=sum(self.enable_lora))

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(
            sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        weight_splitted = weight.split(
            self.qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            # (B, C_output', T)
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)], dim=1
        )  # (B, C_output, T)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""

        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            delta_w = self.conv1d(
                self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
                self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
            ).squeeze(
                0
            )  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
            # W = W + delta_W (merge)
            # (256, 128) after zero_pad (384, 128)
            self.linear.weight.data += self.zero_pad(delta_w * self.scaling)
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
        after_A = F.linear(self.lora_dropout(x), self.lora_A)
        # For F.conv1d:
        # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
        # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
        after_B = self.conv1d(
            after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
            self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).transpose(
            -2, -1
        )  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
        # (64, 64, 256) after zero_pad (64, 64, 384)
        lora = self.zero_pad(after_B) * self.scaling
        return pretrained + lora


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none", freeze_patch_embed: bool = False, freeze_norm: bool = False, free_relative_bias: bool = False, freeze_downsample_reduction=False) -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    def lora_filter(key): return "lora_" in key
    def patch_embed_filter(
        key): return not freeze_patch_embed and "patch_embed" in key

    def norm_filter(key): return not freeze_norm and "norm" in key

    def downsample_reduction_filter(
        key): return not freeze_downsample_reduction and "downsample.reduction" in key

    def relative_position_bias_filter(
        key): return not free_relative_bias and "relative_position_bias_table" in key

    def all_filters(key):
        return lora_filter(key) or patch_embed_filter(key) or norm_filter(key) or downsample_reduction_filter(key) or relative_position_bias_filter(key)

    print(f"LoRA bias mode: {bias}")
    print(f"LoRA Freeze patch_embed: {freeze_patch_embed}")
    print(f"LoRA Freeze norm: {freeze_norm}")
    print(f"LoRA Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f"LoRA Freeze relative_position_bias: {free_relative_bias}")
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if not all_filters(n):
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def mark_prompt_as_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "prompt_embeddings" in name or "deep_prompt_embeddings" in name:
            param.requires_grad = True


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


def merge_lora_weights(model) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str, split_qkv: bool = False) -> Dict:
    unmatched_keys = []
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            weights = state_dict.pop(
                full_checkpoint_name)
            last_four = ".".join(full_attribute_name.split(".")[-4:])
            if split_qkv and last_four in ["attn.qkv.linear.weight", "attn.qkv.linear.bias"]:
                w_q, w_k, w_v = torch.chunk(weights, chunks=3)
                weight_bias = last_four.split(".")[-1]
                full_attribute_name_without_suffix = ".".join(full_attribute_name.split(".")[
                    :-2])
                state_dict[f"{full_attribute_name_without_suffix}.q.linear.{weight_bias}"] = w_q
                state_dict[f"{full_attribute_name_without_suffix}.k.linear.{weight_bias}"] = w_k
                state_dict[f"{full_attribute_name_without_suffix}.v.linear.{weight_bias}"] = w_v
            else:
                state_dict[full_attribute_name] = weights
        else:
            unmatched_keys.append(checkpoint_name)
    if len(unmatched_keys) > 0:
        print(
            f"WARNING: The following keys from the checkpoint were not mapped: {unmatched_keys}")
    return state_dict
