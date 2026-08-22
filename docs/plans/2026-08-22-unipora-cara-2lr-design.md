# UniPoRA-CARA End-to-End 2lr Design

## Goal

让 PoRA-CARA 使用 UniPoRA 已验证的双倍学习率，同时保持旧实验可复现，并确保 Stage1 生成的 resolved config 自动把相同训练超参数传递到正式训练。

## Design

新增 `unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr.yaml`，通过 `BASE` 继承现有 PoRA-CARA 配置，只覆盖 `MODEL.NAME` 与三个学习率字段：`BASE_LR=1e-3`、`WARMUP_LR=1e-6`、`MIN_LR=1e-5`。不修改全局默认值，也不覆盖原配置，避免影响已有输出与其他实验。

Stage1 直接从合并后的配置构建 optimizer，因此会把固定学习率从 `5e-4` 提高到 `1e-3`。Stage1 当前没有 scheduler，`WARMUP_LR` 和 `MIN_LR` 在该阶段不参与计算。Stage1 的 resolved config 会继续把最初传入的 `_2lr.yaml` 记录为 `BASE`；正式训练加载 resolved config 后继承全部三个字段，并继续采用与 UniPoRA 相同的 batch、`WORLD_SIZE` 和 accumulation 学习率缩放公式以及 scheduler。

双进程语义不变：Stage1 两个 rank 独立搜索，正式训练两个 rank 独立训练，rank 0 产物为规范结果。使用新配置必须重新执行 Stage1，不能复用旧的 affinity、grouping 或 `post_affinity_checkpoint.pth`，否则不能称为端到端 2lr pipeline。

## Verification

新增标准库源文件测试，验证配置继承、三个精确数值、独立模型名、Stage1 optimizer 入口和 resolved config 的 `BASE` 传递。随后运行 Stage1 双进程测试、UniPoRA 训练对齐测试、YAML 解析、Python AST 检查和 `git diff --check`。
