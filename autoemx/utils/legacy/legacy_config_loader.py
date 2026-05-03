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

    for key, value in data.items():
        if key not in config_classes_dict:
            metadata[key] = value

    return configs, metadata
