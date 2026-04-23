#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience I/O helpers for sample ledgers."""

from pathlib import Path

from autoemx.config.schemas import SampleLedger


def load_sample_ledger(file_path: str | Path) -> SampleLedger:
    """Load and validate a sample ledger JSON in one call."""
    return SampleLedger.from_json_file(file_path)


def save_sample_ledger(
    ledger: SampleLedger,
    file_path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Save a sample ledger JSON in one call."""
    ledger.to_json_file(file_path, indent=indent)
