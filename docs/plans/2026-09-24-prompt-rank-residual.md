# Prompt Rank Residual Implementation Plan

> Execute the following bounded steps in the current feature branch. The user has authorized improving or redesigning the method; no performance claim is made before training.

**Goal:** Add a testable alternative to diagonal rank masks without changing existing E0/E4 behavior.

**Architecture:** Reuse the routed group's A to encode every post-attention prompt token. Each patch retrieves a centered prompt residual using cosine attention in rank space. A zero-initialized, group-shared rank-to-rank projection writes this residual into the original LoRA path. Start with stage-last fc1 in stages 3 and 4.

**Tech Stack:** Python, PyTorch, YACS, unittest.

---

1. Add behavioral tests in `tests/test_prompt_rank_residual.py`: identity including dropout, equal-mean/different-token conditions, zero-coordinate replenishment, prompt permutation invariance, routed gradients, low-precision finiteness, configuration and checkpoint separation.
2. Add `PromptRankResidual` and an opt-in `RANK_EXTRACT.MODE` in `models/lora.py`, `config.py`, `models/swin_transformer_mtlora.py`, and `models/swin_transformer_vpt.py`. Preserve the default gate and its checkpoint keys. New residual checkpoints use distinct parameter names.
3. Add E4 stage-last and E5 residual experiment configurations. Reuse the existing corrected E0 and the same fresh Stage-1 artifacts. Retain all training hyperparameters.
4. Document evidence, assumptions, alternatives, formulas, cost, ablations and executable server commands in the companion design document and README.
5. Run the focused numerical tests plus prompt/config/training-alignment regressions when dependencies permit. Run syntax and diff checks. Record the exact verification boundary; CPU correctness does not establish dataset performance.
