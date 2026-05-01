#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runner entrypoint for standards schema validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoemx.utils.standards_validator import validate_standards_tree


__all__ = ["run_validate_standards"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate EDS standards JSON files against the schema.",
    )
    parser.add_argument(
        "--root",
        default="autoemx/calibrations",
        help="Root directory to scan for standards files (default: autoemx/calibrations).",
    )
    return parser


def run_validate_standards(root: str = "autoemx/calibrations") -> int:
    """Validate standards files under root and return process-style exit code."""
    root_dir = Path(root)

    if not root_dir.exists():
        print(f"Root directory does not exist: {root_dir}")
        return 2
    if not root_dir.is_dir():
        print(f"Root path is not a directory: {root_dir}")
        return 2

    return validate_standards_tree(root_dir)


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    return run_validate_standards(root=args.root)


if __name__ == "__main__":
    raise SystemExit(main())
