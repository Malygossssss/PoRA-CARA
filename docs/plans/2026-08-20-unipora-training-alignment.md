# UniPoRA Training Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 PoRA-CARA 正式训练逻辑与 UniPoRA 当前实现对齐，同时保留 CARA/AG-MTLoRA 必需流程。

**Architecture:** UniPoRA 的 `main.py`、受约束多任务控制器和 MTL 数据加载语义作为基准；PoRA-CARA 只保留 prompt、group-shared LoRA、Stage-1 和 checkpoint 扩展相关差异。双卡按用户选择采用两个独立进程，不使用 DDP 或分布式 MTL 采样器。

**Tech Stack:** Python 3.8、PyTorch 2.0、YACS、pytest/unittest。

---

### Task 1: Add alignment contract tests

**Files:**
- Create: `tests/test_unipora_training_alignment.py`

**Steps:**
1. 增加源码契约测试，断言正式训练不构造 DDP、没有 `no_sync()`，并保留 prompt 解冻调用。
2. 增加 MTL loader 测试，模拟 world size 2，断言训练/验证 loader 使用完整数据集且不安装 distributed sampler。
3. 运行新测试并确认它在当前 DDP 实现上失败。

### Task 2: Restore UniPoRA training controller

**Files:**
- Modify: `main.py`
- Modify: `config.py`
- Create: `constrained_mtl.py`
- Create: `tests/test_constrained_mtl.py`

**Steps:**
1. 以 UniPoRA `main.py` 为基础恢复训练控制流。
2. 保留 `mark_prompt_as_trainable()` 及 CARA 所需模型入口。
3. 恢复 constrained-MTL/conflict-ratio 配置节点及配置规范化。
4. 引入 UniPoRA 控制器和原测试。

### Task 3: Restore UniPoRA dual-process MTL loading

**Files:**
- Modify: `data/build.py`
- Modify: `data/mtl_ds.py`

**Steps:**
1. 移除 MTL train/val 的 `DistributedSampler`。
2. 恢复训练 `shuffle=True`、验证 `shuffle=False`。
3. 保证 Stage-1 调用签名仍可用。

### Task 4: Document actual launch semantics

**Files:**
- Modify: `README.md`
- Modify: `README_UNIPORA_CARA.md`

**Steps:**
1. 给出与 UniPoRA 一致的双卡启动命令。
2. 明确两个 rank 独立训练、完整读数、只保存 rank 0。

### Task 5: Verify and audit

**Files:**
- Test: `tests/test_unipora_training_alignment.py`
- Test: `tests/test_constrained_mtl.py`
- Test: existing `tests/`

**Steps:**
1. 运行新增定向测试。
2. 运行 Stage-1 与 prompt 回归测试。
3. 运行完整测试和 Python 静态编译。
4. 对比 UniPoRA，确认剩余差异仅属于 CARA/AG-MTLoRA。

