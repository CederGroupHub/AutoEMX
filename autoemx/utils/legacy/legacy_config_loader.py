#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy configuration JSON loader for pre-ledger project layouts."""

import json
from typing import Any, Dict, Tuple

import autoemx.utils.constants as cnst


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
