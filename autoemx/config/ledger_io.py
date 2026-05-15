#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience I/O helpers for sample ledgers."""

from pathlib import Path
from typing import cast

import autoemx.utils.constants as cnst

from autoemx.config.ledger_schemas import LedgerConfigs, SampleLedger # type: ignore


def load_sample_ledger(file_path: str | Path) -> SampleLedger:
    """Load a sample ledger, with full legacy bootstrap when ledger.json is missing.

    If the ledger file does not exist, this uses legacy Data.csv + config JSON files
    to bootstrap a populated ledger (configs, spectra, and quantification payloads).
    """
    ledger_path = Path(file_path)

    if ledger_path.exists():
        ledger = SampleLedger.from_json_file(ledger_path)
        ledger.sample_path = str(ledger_path.parent.resolve())
        print(f"Loaded sample ledger for sample '{ledger.sample_id}' from {ledger_path}")
        return ledger

    sample_result_dir = ledger_path.parent
    from autoemx.utils.legacy.legacy_backfill import load_ledger_configs_from_legacy_json
    from autoemx.utils.legacy.legacy_config_loader import has_legacy_data_csv
    from autoemx.utils.legacy.ledger_bootstrap import ( # type: ignore
        build_legacy_background_pointer_writer, # type: ignore
        build_legacy_import_quantification_config, # type: ignore
        build_legacy_json_pointer_resolver, # type: ignore
        load_or_create_ledger_with_legacy_data_csv, # type: ignore
    )

    legacy_ledger_configs_raw = load_ledger_configs_from_legacy_json(str(sample_result_dir))
    if legacy_ledger_configs_raw is None:
        legacy_config_candidates = [
            sample_result_dir / f"{cnst.CONFIG_FILENAME}.json",
            sample_result_dir / f"{cnst.ACQUISITION_INFO_FILENAME}.json",
        ]
        if any(path.exists() for path in legacy_config_candidates):
            raise FileNotFoundError(
                f"No ledger file found at '{ledger_path}' and legacy config JSON in "
                f"'{sample_result_dir}' could not be parsed into LedgerConfigs."
            )
        raise FileNotFoundError(
            f"No ledger file found at '{ledger_path}' and no legacy config JSON "
            f"was found in '{sample_result_dir}'."
        )
    legacy_ledger_configs = cast(LedgerConfigs, legacy_ledger_configs_raw)

    if has_legacy_data_csv(str(sample_result_dir)):
        legacy_quant_config = build_legacy_import_quantification_config(
            sample_result_dir=str(sample_result_dir),
            ledger_configs=legacy_ledger_configs,
        )
        resolve_or_create_spectrum_pointer = build_legacy_json_pointer_resolver(str(sample_result_dir))
        write_background_pointer = build_legacy_background_pointer_writer(str(sample_result_dir))

        ledger = load_or_create_ledger_with_legacy_data_csv(
            sample_result_dir=str(sample_result_dir),
            sample_id=sample_result_dir.name,
            microscope_id=getattr(legacy_ledger_configs.microscope_cfg, "ID", None),
            use_instrument_background=bool(
                (legacy_quant_config.options or {}).get("use_instrument_background", False)
            ),
            default_ledger_configs=legacy_ledger_configs,
            resolve_or_create_spectrum_pointer=resolve_or_create_spectrum_pointer,
            write_background_pointer=write_background_pointer,
        )
        ledger.sample_path = str(sample_result_dir.resolve())
        print(
            f"Loaded sample ledger for sample '{ledger.sample_id}' from legacy Data.csv/configs "
            f"in '{sample_result_dir}' (ledger.json not found)."
        )
        return ledger

    legacy_quant_config = build_legacy_import_quantification_config(
        sample_result_dir=str(sample_result_dir),
        ledger_configs=legacy_ledger_configs,
    )
    ledger = SampleLedger(
        sample_id=sample_result_dir.name,
        sample_path=str(sample_result_dir.resolve()),
        configs=legacy_ledger_configs,
        spectra=[],
        quantifications=[legacy_quant_config],
        active_quant=int(legacy_quant_config.quantification_id),
    )
    print(
        f"Loaded legacy sample configuration for sample '{ledger.sample_id}' "
        f"from '{sample_result_dir}' (ledger.json and Data.csv not found)."
    )
    return ledger


def save_sample_ledger(
    ledger: SampleLedger,
    file_path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Save a sample ledger JSON in one call."""
    ledger.to_json_file(file_path, indent=indent)
