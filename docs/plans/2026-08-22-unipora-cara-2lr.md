# UniPoRA-CARA End-to-End 2lr Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible PoRA-CARA configuration that applies UniPoRA's validated 2x learning rates to Stage1 and formal training.

**Architecture:** Keep the existing configuration immutable and add a minimal inherited `_2lr` variant. Stage1 consumes `BASE_LR` directly; its generated resolved config points back to the `_2lr` file so formal training inherits all scheduler learning-rate fields.

**Tech Stack:** YAML/YACS configuration, Python `unittest`, existing Stage1 and formal training launchers.

---

### Task 1: Add the failing configuration contract test

**Files:**
- Create: `tests/test_unipora_cara_2lr_config.py`

**Step 1: Write the failing test**

Test that the new config exists, inherits the original config, uses a distinct model name, and contains exactly:

```yaml
TRAIN:
  BASE_LR: 1.0e-3
  WARMUP_LR: 1.0e-6
  MIN_LR: 1.0e-5
```

Also assert that Stage1 builds its optimizer from the runtime config and that resolved configs retain the original `base_cfg_path`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -p test_unipora_cara_2lr_config.py -v`

Expected: failure because the `_2lr.yaml` file does not exist.

### Task 2: Add the inherited 2lr config

**Files:**
- Create: `configs/mtlora/tiny_448/pascal/unipora_cara_tiny_448_r64_prom50_global_group_proxy_2lr.yaml`

**Step 1: Implement the minimal config**

Inherit the original PoRA-CARA config, set a distinct `_2lr` model name, and override the three learning-rate fields.

**Step 2: Run the focused test**

Run: `python -m unittest discover -s tests -p test_unipora_cara_2lr_config.py -v`

Expected: all tests pass.

### Task 3: Update workflow documentation

**Files:**
- Modify: `README_UNIPORA_CARA.md`
- Modify: `README.md`

**Step 1: Update recommended commands**

Use the `_2lr.yaml` file in the end-to-end UniPoRA-CARA workflow and explain Stage1 versus formal-training LR behavior.

**Step 2: Document rerun requirements**

State that old Stage1 artifacts must not be reused for an end-to-end 2lr result.

### Task 4: Verify and commit

**Files:**
- Test: `tests/test_unipora_cara_2lr_config.py`
- Test: `tests/test_stage1_multiprocess.py`
- Test: `tests/test_unipora_training_alignment.py`

**Step 1: Run focused test suites**

Expected: all source-based tests pass.

**Step 2: Parse YAML and Python sources**

Expected: the new config resolves to the exact three values; all Python files pass AST parsing.

**Step 3: Check patch hygiene**

Run: `git diff --check`

Expected: no whitespace errors.

**Step 4: Commit**

Commit message: `Add end-to-end 2lr UniPoRA-CARA config`
