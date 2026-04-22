#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience I/O helpers for sample ledgers."""

import json
from pathlib import Path
from typing import Literal

from autoemxsp.config.schemas import LedgerConfigs, SampleLedger


def load_sample_ledger(file_path: str | Path) -> SampleLedger:
    """Load and validate a sample ledger JSON in one call."""
    return SampleLedger.from_json_file(file_path)


def save_sample_ledger(
    ledger: SampleLedger,
    file_path: str | Path,
    *,
    config_key_style: Literal["cfg", "config"] = "cfg",
    indent: int = 2,
) -> None:
    """Save a sample ledger JSON in one call."""
    ledger.to_json_file(file_path, config_key_style=config_key_style, indent=indent)


def load_split_ledger(
    ledger_path: str | Path,
    configs_path: str | Path,
    *,
    config_key_style: Literal["cfg", "config"] = "cfg",
) -> SampleLedger:
    """Load ledger and configs from separate files and merge into one model."""
    ledger = SampleLedger.from_json_file(ledger_path)
    configs = LedgerConfigs.model_validate_json(Path(configs_path).read_text(encoding="utf-8"))
    merged_payload = ledger.model_dump(mode="json")
    cfg_payload = configs.model_dump(mode="json")

    if config_key_style == "config":
        cfg_payload = {
            (k[: -len("_cfg")] + "_config") if k.endswith("_cfg") else k: v
            for k, v in cfg_payload.items()
        }

    merged_payload["configs"] = cfg_payload
    return SampleLedger.model_validate(merged_payload)


def save_split_ledger(
    ledger: SampleLedger,
    ledger_path: str | Path,
    configs_path: str | Path,
    *,
    config_key_style: Literal["cfg", "config"] = "cfg",
    indent: int = 2,
) -> None:
    """Save ledger and configs into separate files (recommended standard)."""
    if ledger.configs is None:
        raise ValueError("SampleLedger.configs is None; cannot write split configs file")

    ledger.to_json_file(
        ledger_path,
        config_key_style=config_key_style,
        include_configs=False,
        indent=indent,
    )

    cfg_payload = ledger.configs.model_dump(mode="json")
    if config_key_style == "config":
        cfg_payload = {
            (k[: -len("_cfg")] + "_config") if k.endswith("_cfg") else k: v
            for k, v in cfg_payload.items()
        }

    cfg_path = Path(configs_path)
    cfg_path.write_text(
        json.dumps(cfg_payload, indent=indent) + "\n",
        encoding="utf-8",
    )
