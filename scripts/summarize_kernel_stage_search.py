#!/usr/bin/env python3
"""Aggregate raw four-task metrics from paired epoch-299 artifacts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_stage_search.reporting import summarize_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-manifest", required=True)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize_files(args.formal_manifest, args.endpoints, args.output)
    print(f"Summarized {result['unique_training_runs']} unique runs at epoch 299: {args.output}")


if __name__ == "__main__":
    main()
