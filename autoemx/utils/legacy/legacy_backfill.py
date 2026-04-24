#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy compatibility helpers for reconstructing spectra pointer files from Data.csv."""

import ast
import json
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


def backfill_spectra_from_data_csv(
    data_csv_path: str,
    resolve_or_create_pointer: Callable[..., str],
    *,
    spectrum_key: str,
    spectrum_id_key: str,
    live_time_key: str,
    real_time_key: str,
    background_key: Optional[str] = None,
    write_background_pointer: Optional[Callable[..., Optional[str]]] = None,
) -> int:
    """Reconstruct spectra pointer files from legacy Data.csv content.

    The callback must accept: spectrum_id, spectrum_vals, live_time, real_time.
    Optionally, background arrays can be persisted via write_background_pointer.
    """
    if not data_csv_path:
        return 0

    try:
        data_df = pd.read_csv(data_csv_path)
    except Exception:
        return 0

    if spectrum_key not in data_df.columns:
        return 0

    n_written = 0
    for row_idx, row in data_df.iterrows():
        spectrum_raw = row.get(spectrum_key)
        if pd.isna(spectrum_raw):
            continue

        try:
            spectrum_vals = ast.literal_eval(spectrum_raw) if isinstance(spectrum_raw, str) else spectrum_raw
        except Exception:
            continue

        if not isinstance(spectrum_vals, (list, tuple, np.ndarray)):
            continue

        spectrum_id_val = row.get(spectrum_id_key, row_idx)
        if isinstance(spectrum_id_val, (int, np.integer, float)) and not pd.isna(spectrum_id_val):
            spectrum_id = str(int(spectrum_id_val))
        else:
            spectrum_id = str(spectrum_id_val)
        if not spectrum_id:
            spectrum_id = str(row_idx)

        live_time = row.get(live_time_key)
        real_time = row.get(real_time_key)
        live_time = None if pd.isna(live_time) else float(live_time)
        real_time = None if pd.isna(real_time) else float(real_time)
        # Legacy Data.csv files may only carry real_time; treat it as collection time.
        collection_time = live_time if live_time is not None else real_time

        pointer_relpath = resolve_or_create_pointer(
            spectrum_id=spectrum_id,
            spectrum_vals=list(map(float, spectrum_vals)),
            live_time=collection_time,
            real_time=real_time,
        )
        if pointer_relpath:
            n_written += 1

        if background_key and write_background_pointer is not None:
            background_raw = row.get(background_key)
            if pd.isna(background_raw):
                continue
            try:
                background_vals = (
                    ast.literal_eval(background_raw)
                    if isinstance(background_raw, str)
                    else background_raw
                )
            except Exception:
                continue
            if not isinstance(background_vals, (list, tuple, np.ndarray)):
                continue
            write_background_pointer(
                spectrum_id=spectrum_id,
                background_vals=list(map(float, background_vals)),
            )

    return n_written


def load_ledger_configs_from_legacy_json(sample_result_dir: str) -> Optional[object]:
    """Try to reconstruct a LedgerConfigs from a legacy config JSON in the sample directory.

    Checks for ``config.json`` first, then ``Comp_analysis_configs.json``. Returns a
    ``LedgerConfigs`` instance on success, or ``None`` if neither file is found / parseable.

    Imports are deferred to avoid circular dependencies.
    """
    import autoemx.utils.constants as cnst
    from autoemx.config.classes import config_classes_dict
    from autoemx.config.schemas import LedgerConfigs

    candidate_files = [
        os.path.join(sample_result_dir, f"{cnst.CONFIG_FILENAME}.json"),
        os.path.join(sample_result_dir, f"{cnst.ACQUISITION_INFO_FILENAME}.json"),
    ]

    raw: Optional[dict] = None
    for path in candidate_files:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                break
            except Exception:
                continue

    if raw is None:
        return None

    # Map each config key to its Pydantic model, tolerating missing/extra fields
    cfg_kwargs: dict = {}
    for cfg_key, cfg_class in config_classes_dict.items():
        if cfg_key not in raw:
            continue
        try:
            cfg_kwargs[cfg_key] = cfg_class.model_validate(raw[cfg_key])
        except Exception:
            # Field may be missing or schema changed — skip this config block
            continue

    # LedgerConfigs requires the four mandatory configs; if any are absent, abort
    required = {
        cnst.MICROSCOPE_CFG_KEY,
        cnst.SAMPLE_CFG_KEY,
        cnst.MEASUREMENT_CFG_KEY,
        cnst.SAMPLESUBSTRATE_CFG_KEY,
    }
    if not required.issubset(cfg_kwargs):
        return None

    # Build LedgerConfigs, mapping constant keys to field names
    key_to_field = {
        cnst.MICROSCOPE_CFG_KEY: "microscope_cfg",
        cnst.SAMPLE_CFG_KEY: "sample_cfg",
        cnst.MEASUREMENT_CFG_KEY: "measurement_cfg",
        cnst.SAMPLESUBSTRATE_CFG_KEY: "sample_substrate_cfg",
        cnst.QUANTIFICATION_CFG_KEY: "quant_cfg",
        cnst.CLUSTERING_CFG_KEY: "clustering_cfg",
        cnst.PLOT_CFG_KEY: "plot_cfg",
        cnst.POWDER_MEASUREMENT_CFG_KEY: "powder_meas_cfg",
        cnst.BULK_MEASUREMENT_CFG_KEY: "bulk_meas_cfg",
        cnst.EXP_STD_MEASUREMENT_CFG_KEY: "exp_stds_cfg",
    }
    ledger_cfg_kwargs = {
        field: cfg_kwargs[key]
        for key, field in key_to_field.items()
        if key in cfg_kwargs
    }

    try:
        return LedgerConfigs(**ledger_cfg_kwargs)
    except Exception:
        return None