#!/usr/bin/env python3
"""Opt-in synthetic integration or real-geometry structural smoke tests."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--structure", action="store_true", help="448 random-weight backbone smoke only; no real labels or calibration")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from kernel_stage_search.synthetic import check_synthetic, check_real_structure
    result = check_synthetic(args.device, args.output_dir) if args.synthetic else check_real_structure(args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
