# Stage-1 Numerical Stability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Stage-1 numerically aligned with formal training and fail with actionable persisted diagnostics before an invalid grouping checkpoint can reach Stage-2.

**Architecture:** Extract one learning-rate scaling helper used by both launchers, then give Stage-1 a single optimizer/scheduler/scaler path for warmup and affinity epochs. Add safe ignored-label loss handling, finite-value guards, atomic last-good checkpoints, an AMP compatibility smoke test, and launcher-level failure manifests.

**Tech Stack:** Python, PyTorch AMP, timm cosine scheduler, YACS configuration, unittest/mock, JSON manifests.

---

### Task 1: Lock the learning-rate and ignored-label contracts

**Files:**
- Modify: `tests/test_unipora_training_alignment.py`
- Modify: `tests/test_ag_mtlora_stage1.py`
- Modify: `mtl_loss_schemes.py`
- Modify: `utils.py`
- Modify: `main.py`
- Modify: `scripts/ag_mtlora_stage1_prepare.py`

**Step 1: Write failing tests**

Add a pure helper test asserting that raw `1e-3` with batch 9 and world size 1 becomes `1.7578125e-5`, and an idempotence guard test that rejects a second scale call. Add a PyTorch test asserting that an all-255 target returns finite differentiable zero from `SoftMaxwithLoss`.

**Step 2: Run focused tests and verify failure**

Run: `python -m unittest tests.test_unipora_training_alignment tests.test_ag_mtlora_stage1 -v`

Expected: new helper and safe-loss tests fail before implementation.

**Step 3: Implement minimal shared behavior**

Add `scale_learning_rates(config, world_size)` in `utils.py`, mutate the three runtime LR fields once, and record a runtime-only guard attribute. Replace the duplicated block in `main.py`; call the helper in the Stage-1 launcher before the optimizer is constructed. In `SoftMaxwithLoss.forward`, return `out.sum() * 0.0` when `(label != ignore_index).any()` is false.

**Step 4: Run focused tests**

Expected: LR and all-ignore contracts pass.

### Task 2: Align the Stage-1 optimizer path

**Files:**
- Modify: `ag_mtlora/stage1.py`
- Modify: `tests/test_ag_mtlora_stage1.py`

**Step 1: Write failing scheduler/scaler tests**

Use small mock models/loaders to assert one continuous scheduler covers warmup plus affinity steps, `step_update` is called once per optimizer update, autocast follows `config.AMP_ENABLE`, and gradient clipping uses `config.TRAIN.CLIP_GRAD`.

**Step 2: Implement the training runtime**

Create a Stage-1 runtime object containing optimizer, cloned scheduler config, scheduler, GradScaler wrapper, global step and AMP-enabled flag. Set scheduler epochs to `warmup_epochs + affinity_score_epochs` and scheduler warmup epochs to `warmup_epochs`. Route both warmup and affinity optimizer updates through one checked step function.

**Step 3: Verify focused tests**

Expected: scheduler step counts, scaler path and clip value pass.

### Task 3: Add numerical guards and final AMP smoke test

**Files:**
- Modify: `ag_mtlora/stage1.py`
- Modify: `tests/test_ag_mtlora_stage1.py`

**Step 1: Write failing non-finite tests**

Test finite scalar/tensor helpers, first-nonfinite state lookup, contextual numerical exceptions, and restoration of model buffers after smoke testing.

**Step 2: Guard Stage-1 operations**

Before affinity dot products, reject non-finite task losses and flattened gradients. Before and after optimizer steps, reject non-finite total loss/grad norm/loss scale. At successful epoch boundaries scan model parameters and buffers.

**Step 3: Implement AMP smoke test**

Snapshot the pre-smoke state, run one train-mode batch under the same autocast setting, validate outputs/loss/gradients, and restore the exact state in `finally`. Only save the post-affinity checkpoint after this passes.

**Step 4: Verify focused tests**

Expected: injected NaN/Inf errors identify phase, task and tensor; smoke test restoration passes.

### Task 4: Persist last-good and failure diagnostics

**Files:**
- Modify: `ag_mtlora/stage1.py`
- Modify: `ag_mtlora/stage1_multiprocess.py`
- Modify: `scripts/ag_mtlora_stage1_prepare.py`
- Modify: `tests/test_stage1_multiprocess.py`
- Modify: `tests/test_ag_mtlora_stage1.py`

**Step 1: Write failing artifact tests**

Assert atomic `last_good_checkpoint.pth` replacement after a successful epoch, structured failure report fields, rank-specific filenames, `status: failed`, and success/failure terminal markers.

**Step 2: Implement diagnostic persistence**

Add atomic JSON and torch-save helpers. Preserve a compact mutable execution context throughout the pipeline. At launcher scope catch exceptions, log `STAGE1_ABORTED` with traceback, write the failure report and failed rank manifest, flush logging handlers, then re-raise for a nonzero exit. Log `STAGE1_COMPLETED` only after all canonical artifacts and manifests exist.

**Step 3: Verify artifact tests**

Expected: success and injected failure paths produce mutually exclusive terminal state.

### Task 5: Documentation and repository verification

**Files:**
- Modify: `README_UNIPORA_CARA.md`
- Modify: `docs/plans/2026-08-22-unipora-cara-2lr-design.md`
- Test: `tests/test_ag_mtlora_stage1.py`
- Test: `tests/test_stage1_multiprocess.py`
- Test: `tests/test_unipora_training_alignment.py`
- Test: `tests/test_unipora_cara_2lr_config.py`

**Step 1: Correct workflow documentation**

Remove the obsolete claim that Stage-1 intentionally consumes unscaled `BASE_LR`. Document invalidation of the August 22 artifacts, success criteria, failure files and the requirement to rerun Stage-1 from the original Swin backbone.

**Step 2: Run focused and static verification**

Run the four focused unittest modules in a PyTorch environment. Locally run `python -m py_compile` on modified Python files, parse YAML, and run `git diff --check`.

Expected: all available tests pass, unavailable GPU/PyTorch execution is explicitly reported, no syntax or whitespace errors remain.

**Step 3: Review final diff**

Confirm no existing experiment output is modified and no invalid Stage-1 artifact is reused.
