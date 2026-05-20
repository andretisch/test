#!/usr/bin/env python3
"""CLI entry point for vehicle counting pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import VehicleCounterPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Подсчёт ТС по линии (CLI)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: config/default.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source = config.resolve_path(config.source_video)
    target = config.resolve_path(config.target_video)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        print(f"Source video not found: {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {source} -> {target}")
    pipeline = VehicleCounterPipeline(config)
    result = pipeline.run(
        source_path=source,
        target_path=target,
        line_start=config.line.start,
        line_end=config.line.end,
    )

    print(f"Done: {result.target_path}")
    print(f"  {config.labels.in_label}: {result.in_count}")
    print(f"  {config.labels.out_label}: {result.out_count}")

    counts_path = Path(result.target_path).with_suffix(".json")
    print(f"Counts saved: {counts_path}")


if __name__ == "__main__":
    main()
