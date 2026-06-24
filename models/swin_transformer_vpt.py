#!/usr/bin/env python3
"""Prompt-aware Swin Transformer for CARA's AG-MTLoRA path.

Only the minimal static deep prepend prompt variant is supported here.
"""

import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from torch.nn import Dropout

from timm.models.layers import to_2tuple

from .swin_transformer_mtlora import (
    PatchMerging,
    SwinTransformerBlock,
    SwinTransformerMTLoRA,
    WindowAttention,
    window_partition,
    window_reverse,
)


class PromptedSwinTransformer(SwinTransformerMTLoRA):
    def __init__(
        self,
        prompt_config,
        tasks,
        mtlora,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        _validate_prompt_config(prompt_config)
        if not tasks:
            raise ValueError("PromptedSwinTransformer requires at least one task.")

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            fused_window_process=kwargs.get("fused_window_process", False),
            tasks=tasks,
            mtlora=mtlora,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PromptedBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PromptedPatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                num_prompts=prompt_config.NUM_TOKENS,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=i_layer,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.prompt_config = prompt_config
        self.tasks = list(tasks)
        self.mtlora = mtlora
        self.num_tokens = int(prompt_config.NUM_TOKENS)
        self.prompt_dropout = Dropout(float(prompt_config.DROPOUT))
        self.prompt_embeddings = nn.ParameterDict()
        self.deep_prompt_embeddings = nn.ModuleList()

        patch_size = to_2tuple(patch_size)
        val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + embed_dim))
        for task in self.tasks:
            emb = torch.zeros(1, self.num_tokens, embed_dim)
            nn.init.uniform_(emb, -val, val)
            self.prompt_embeddings[task] = nn.Parameter(emb)

        for layer_idx, depth in enumerate(depths):
            stage_dim = embed_dim * (2 ** layer_idx)
            num_prompt_blocks = max(depth - 1, 0) if layer_idx == 0 else depth
            prompt_dict = nn.ParameterDict()
            for task in self.tasks:
                emb = torch.zeros(num_prompt_blocks, self.num_tokens, stage_dim)
                if emb.numel() > 0:
                    nn.init.uniform_(emb, -val, val)
                prompt_dict[task] = nn.Parameter(emb)
            self.deep_prompt_embeddings.append(prompt_dict)

    def forward(self, x, task=None, return_stages=False, flatten_ft=False):
        if task is None:
            raise ValueError("PromptedSwinTransformer.forward requires a task.")
        x = self.forward_features(x, task, return_stages)
        if return_stages:
            return x
        x = self.head(x)
        if flatten_ft:
            return x
        return x

    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        return self.pos_drop(x)

    def incorporate_prompt(self, x, task):
        if task not in self.prompt_embeddings:
            raise KeyError(f"Unknown prompt task: {task}")
        x = self.get_patch_embeddings(x)
        prompt_embd = self.prompt_dropout(
            self.prompt_embeddings[task].expand(x.shape[0], -1, -1)
        )
        return torch.cat((prompt_embd, x), dim=1)

    def forward_features(self, x, task, return_stages=False):
        x = self.incorporate_prompt(x, task)
        feats = []
        for layer_idx, layer in enumerate(self.layers):
            deep_prompt_embd = self.prompt_dropout(
                self.deep_prompt_embeddings[layer_idx][task]
            )
            x = layer(x, task, deep_prompt_embd)
            if return_stages:
                prompt_count = x.shape[1] - (
                    layer.output_resolution[0] * layer.output_resolution[1]
                )
                feats.append(x[:, prompt_count:, :])

        if self.norm is not None:
            x = self.norm(x)
        if return_stages:
            return feats
        final_prompt_count = x.shape[1] - (
            self.layers[-1].output_resolution[0] * self.layers[-1].output_resolution[1]
        )
        x = x[:, final_prompt_count:, :]
        x = self.avgpool(x.transpose(1, 2))
        return torch.flatten(x, 1)


class PromptedBasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        num_prompts=None,
        tasks=None,
        mtlora=None,
        layer_idx=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = (
            (input_resolution[0] // 2, input_resolution[1] // 2)
            if downsample is not None
            else input_resolution
        )
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.tasks = tasks
        self.num_prompts = num_prompts

        self.blocks = nn.ModuleList([
            PromptedSwinTransformerBlock(
                num_prompts,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                tasks=tasks,
                mtlora=mtlora,
                layer_idx=layer_idx,
                lora=(i == depth - 1),
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                num_prompts,
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                layer_idx=layer_idx,
                mtlora=mtlora,
            )
        else:
            self.downsample = None

    def forward(self, x, task, deep_prompt_embd):
        B = x.shape[0]
        num_blocks = len(self.blocks)
        if deep_prompt_embd.shape[0] not in {max(num_blocks - 1, 0), num_blocks}:
            raise ValueError(
                "Deep prompt block count must equal depth or depth - 1 for the first stage."
            )

        if deep_prompt_embd.shape[0] == num_blocks:
            for i in range(num_blocks):
                x = self._replace_prompt(x, deep_prompt_embd[i].expand(B, -1, -1))
                x, tasks_lora = self.blocks[i](x)
                if tasks_lora is not None:
                    x = tasks_lora[task]
        else:
            for i in range(num_blocks):
                if i > 0:
                    x = self._replace_prompt(x, deep_prompt_embd[i - 1].expand(B, -1, -1))
                x, tasks_lora = self.blocks[i](x)
                if tasks_lora is not None:
                    x = tasks_lora[task]

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def _replace_prompt(self, x, prompt_emb):
        prompt_count = x.shape[1] - (self.input_resolution[0] * self.input_resolution[1])
        return torch.cat((prompt_emb, x[:, prompt_count:, :]), dim=1)


class PromptedPatchMerging(PatchMerging):
    def __init__(
        self,
        num_prompts,
        input_resolution,
        dim,
        norm_layer=nn.LayerNorm,
        layer_idx=0,
        mtlora=None,
    ):
        super().__init__(input_resolution, dim, norm_layer, layer_idx=layer_idx, mtlora=mtlora)
        self.num_prompts = num_prompts

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        prompt_count = L - (H * W)
        prompt_emb = x[:, :prompt_count, :]
        x = x[:, prompt_count:, :]
        assert x.shape[1] == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        prompt_emb = torch.cat((prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = torch.cat((prompt_emb, x), dim=1)
        x = self.norm(x)
        x, _ = self.reduction(x)
        return x


class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self,
        num_prompts,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        tasks=None,
        mtlora=None,
        layer_idx=0,
        lora=False,
    ):
        super().__init__(
            dim,
            input_resolution,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
            fused_window_process=False,
            lora=lora,
            tasks=tasks,
            mtlora=mtlora,
            layer_idx=layer_idx,
        )
        self.num_prompts = num_prompts
        self.attn = PromptedWindowAttention(
            num_prompts,
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            tasks=tasks,
            mtlora=mtlora,
            layer_idx=layer_idx,
            lora=lora,
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        prompt_count = L - (H * W)
        prompt_emb = x[:, :prompt_count, :]
        x = x[:, prompt_count:, :]
        assert x.shape[1] == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        num_windows = int(x_windows.shape[0] / B)
        prompt_windows = prompt_emb.unsqueeze(0).expand(num_windows, -1, -1, -1)
        prompt_windows = prompt_windows.reshape((-1, prompt_count, C))
        x_windows = torch.cat((prompt_windows, x_windows), dim=1)

        attn_windows, attn_windows_lora_tasks = self.attn(x_windows, mask=self.attn_mask)

        prompt_emb = attn_windows[:, :prompt_count, :]
        attn_windows = attn_windows[:, prompt_count:, :]
        prompt_emb = prompt_emb.view(-1, B, prompt_count, C).mean(0)

        prompt_emb_lora_tasks = {}
        if attn_windows_lora_tasks is not None:
            for task in self.tasks:
                prompt_task = attn_windows_lora_tasks[task][:, :prompt_count, :]
                prompt_emb_lora_tasks[task] = prompt_task.view(-1, B, prompt_count, C).mean(0)
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task][:, prompt_count:, :]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = x.view(B, H * W, C)

        if attn_windows_lora_tasks is not None:
            for task in self.tasks:
                task_windows = attn_windows_lora_tasks[task].view(-1, self.window_size, self.window_size, C)
                task_x = window_reverse(task_windows, self.window_size, H, W)
                if self.shift_size > 0:
                    task_x = torch.roll(task_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                task_x = task_x.view(B, H * W, C)
                attn_windows_lora_tasks[task] = torch.cat(
                    (prompt_emb_lora_tasks[task], task_x), dim=1
                )
                attn_windows_lora_tasks[task] = shortcut + self.drop_path(
                    attn_windows_lora_tasks[task]
                )

        x = torch.cat((prompt_emb, x), dim=1)
        x = shortcut + self.drop_path(x)

        mlp_inputs = (
            {task: self.norm2(attn_windows_lora_tasks[task]) for task in self.tasks}
            if attn_windows_lora_tasks is not None
            else None
        )
        mlp_result, mlp_lora_tasks = self.mlp(self.norm2(x), mlp_inputs)
        if mlp_lora_tasks is None:
            return x + self.drop_path(mlp_result), None

        if attn_windows_lora_tasks is None:
            for task in self.tasks:
                mlp_lora_tasks[task] = shortcut + self.drop_path(mlp_lora_tasks[task])
        else:
            for task in self.tasks:
                mlp_lora_tasks[task] = attn_windows_lora_tasks[task] + self.drop_path(
                    mlp_lora_tasks[task]
                )
        return x + self.drop_path(mlp_result), mlp_lora_tasks


class PromptedWindowAttention(WindowAttention):
    def __init__(
        self,
        num_prompts,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        tasks=None,
        mtlora=None,
        layer_idx=0,
        lora=False,
    ):
        super().__init__(
            dim,
            window_size,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            lora=lora,
            tasks=tasks,
            mtlora=mtlora,
            layer_idx=layer_idx,
        )
        self.num_prompts = num_prompts

    def _apply_attention_from_qkv(self, qkv, B_, N, C, mask=None):
        qkv = self._reshape_qkv(qkv, B_, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        window_tokens = self.window_size[0] * self.window_size[1]
        prompt_count = N - window_tokens
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(window_tokens, window_tokens, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        if prompt_count > 0:
            padded_bias = relative_position_bias.new_zeros(
                self.num_heads, N, N
            )
            padded_bias[:, prompt_count:, prompt_count:] = relative_position_bias
            relative_position_bias = padded_bias
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            if prompt_count > 0:
                padded_mask = mask.new_zeros(nW, N, N)
                padded_mask[:, prompt_count:, prompt_count:] = mask
                mask = padded_mask
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        return (attn @ v).transpose(1, 2).reshape(B_, N, C)

    def forward(self, x, x_tasks=None, mask=None):
        B_, N, C = x.shape
        qkv, qkv_tasks = self.qkv(x, x_tasks)
        x = self._apply_attention_from_qkv(qkv, B_, N, C, mask=mask)

        attn_task_outputs = None
        if qkv_tasks is not None:
            attn_task_outputs = {
                task: self._apply_attention_from_qkv(qkv_tasks[task], B_, N, C, mask=mask)
                for task in self.tasks
            }

        x, x_proj_lora_tasks = self.proj(x, attn_task_outputs)
        x = self.proj_drop(x)
        if x_proj_lora_tasks is not None:
            for task in self.tasks:
                x_proj_lora_tasks[task] = self.proj_drop(x_proj_lora_tasks[task])
        return x, x_proj_lora_tasks


def _validate_prompt_config(prompt_config):
    if str(prompt_config.LOCATION) != "prepend":
        raise ValueError("CARA prompt support currently only allows MODEL.PROMPT.LOCATION='prepend'.")
    if not bool(prompt_config.DEEP):
        raise ValueError("CARA prompt support currently requires MODEL.PROMPT.DEEP=True.")
    if int(prompt_config.NUM_TOKENS) <= 0:
        raise ValueError("MODEL.PROMPT.NUM_TOKENS must be > 0 when prompt is enabled.")
    if str(prompt_config.INITIATION) != "random":
        raise ValueError("CARA prompt support currently only allows MODEL.PROMPT.INITIATION='random'.")
