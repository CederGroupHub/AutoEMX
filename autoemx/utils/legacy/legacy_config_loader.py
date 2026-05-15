#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy configuration JSON loader for pre-ledger project layouts."""

import json
import os
from typing import Any, Dict, Tuple

import autoemx.utils.constants as cnst
from autoemx.config.runtime_configs import (
    BulkMeasurementConfig,
    ExpStandardsConfig,
    MeasurementConfig,
    PowderMeasurementConfig,
)


def get_legacy_data_csv_path(sample_dir: str, data_filename: str = cnst.DATA_FILENAME) -> str:
    """Return the legacy CSV path for a sample directory and base filename."""
    return os.path.join(sample_dir, f"{data_filename}{cnst.DATA_FILEEXT}")


def has_legacy_data_csv(sample_dir: str, data_filename: str = cnst.DATA_FILENAME) -> bool:
    """Return whether the legacy Data.csv-like file exists for a sample directory."""
    return os.path.exists(get_legacy_data_csv_path(sample_dir, data_filename=data_filename))


def load_legacy_configurations_from_json(
    json_path: str,
    config_classes_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Rebuild runtime config objects from a legacy config JSON payload.

    This function exists only for fallback paths where no ledger is available yet.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    configs: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for key, cls in config_classes_dict.items():
        if key in data:
            if isinstance(data[key], dict):
                # Drop legacy keys that are no longer part of strict runtime schemas.
                data[key].pop("num_CPU_cores", None)
                if key == cnst.SAMPLE_CFG_KEY:
                    data[key].pop("ID", None)
            configs[key] = cls(**data[key])
        else:
            configs[key] = None

    measurement_cfg = configs.get(cnst.MEASUREMENT_CFG_KEY)
    if measurement_cfg is None:
        measurement_cfg = MeasurementConfig()

    if not isinstance(measurement_cfg, MeasurementConfig):
        measurement_cfg = MeasurementConfig.model_validate(measurement_cfg)

    raw_acq_payload = data.get(cnst.ACQUISITION_CFG_KEY)
    if not isinstance(raw_acq_payload, dict):
        raw_acq_payload = {}

    powder_cfg_obj = None
    bulk_cfg_obj = None
    exp_cfg_obj = None

    raw_powder_cfg = data.get(cnst.POWDER_MEASUREMENT_CFG_KEY)
    if not isinstance(raw_powder_cfg, dict):
        raw_powder_cfg = raw_acq_payload.get(cnst.POWDER_MEASUREMENT_CFG_KEY)
    if isinstance(raw_powder_cfg, dict):
        try:
            powder_cfg_obj = PowderMeasurementConfig.model_validate(raw_powder_cfg)
        except Exception:
            powder_cfg_obj = None

    raw_bulk_cfg = data.get(cnst.BULK_MEASUREMENT_CFG_KEY)
    if not isinstance(raw_bulk_cfg, dict):
        raw_bulk_cfg = raw_acq_payload.get(cnst.BULK_MEASUREMENT_CFG_KEY)
    if isinstance(raw_bulk_cfg, dict):
        try:
            bulk_cfg_obj = BulkMeasurementConfig.model_validate(raw_bulk_cfg)
        except Exception:
            bulk_cfg_obj = None

    raw_exp_cfg = data.get(cnst.EXP_STD_MEASUREMENT_CFG_KEY)
    if not isinstance(raw_exp_cfg, dict):
        raw_exp_cfg = raw_acq_payload.get(cnst.EXP_STD_MEASUREMENT_CFG_KEY)
    if isinstance(raw_exp_cfg, dict):
        try:
            exp_cfg_obj = ExpStandardsConfig.model_validate(raw_exp_cfg)
        except Exception:
            exp_cfg_obj = None

    saved_images_extension = raw_acq_payload.get("saved_images_extension")
    save_raw_images = raw_acq_payload.get("save_raw_images")
    update_payload: Dict[str, Any] = {
        "powder_meas_cfg": powder_cfg_obj,
        "bulk_meas_cfg": bulk_cfg_obj,
        "exp_stds_cfg": exp_cfg_obj,
    }
    if saved_images_extension is not None:
        update_payload["saved_images_extension"] = saved_images_extension
    if save_raw_images is not None:
        update_payload["save_raw_images"] = save_raw_images

    configs[cnst.MEASUREMENT_CFG_KEY] = measurement_cfg.model_copy(update=update_payload)
    configs[cnst.POWDER_MEASUREMENT_CFG_KEY] = configs[cnst.MEASUREMENT_CFG_KEY].powder_meas_cfg
    configs[cnst.BULK_MEASUREMENT_CFG_KEY] = configs[cnst.MEASUREMENT_CFG_KEY].bulk_meas_cfg
    configs[cnst.EXP_STD_MEASUREMENT_CFG_KEY] = configs[cnst.MEASUREMENT_CFG_KEY].exp_stds_cfg

    # Legacy compatibility:
    # ClusteringConfig is not part of runtime config_classes_dict, but legacy
    # Comp_analysis_configs.json files do include a clustering_cfg payload.
    # Recover it explicitly so first-run analysis (before ledger exists) uses
    # the intended ref_formulae and clustering options.
    if cnst.CLUSTERING_CFG_KEY in data and configs.get(cnst.CLUSTERING_CFG_KEY) is None:
        # Deferred import avoids circular imports during schema module initialization.
        from autoemx.config.ledger_schemas import ClusteringConfig

        raw_clustering_cfg = data.get(cnst.CLUSTERING_CFG_KEY)
        if isinstance(raw_clustering_cfg, dict):
            normalized_payload = dict(raw_clustering_cfg)

            # Legacy key name
            if "k_forced" not in normalized_payload and "k" in normalized_payload:
                normalized_payload["k_forced"] = normalized_payload.get("k")
            normalized_payload.pop("k", None)

            # Keep only schema-declared fields to avoid extra-field errors.
            allowed_fields = set(ClusteringConfig.model_fields.keys())
            normalized_payload = {
                key: value
                for key, value in normalized_payload.items()
                if key in allowed_fields
            }

            try:
                configs[cnst.CLUSTERING_CFG_KEY] = ClusteringConfig.model_validate(normalized_payload)
            except Exception:
                configs[cnst.CLUSTERING_CFG_KEY] = None
        else:
            configs[cnst.CLUSTERING_CFG_KEY] = None

    for key, value in data.items():
        if key not in config_classes_dict:
            metadata[key] = value

    return configs, metadata
