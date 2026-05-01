#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy compatibility helpers for standards-library JSON payloads."""

from __future__ import annotations

from typing import Any, Dict


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def is_legacy_standards_payload(payload: Any) -> bool:
    """Return True when payload matches the legacy top-level format.

    Legacy standards files store measurement modes directly at the root, e.g.
    ``{"point": {...}, "map": {...}}``.
    """
    return _is_mapping(payload) and "standards_by_mode" not in payload


def migrate_legacy_standards_payload(
    payload: Dict[str, Any],
    measurement_type: str,
    beam_energy_keV: int,
) -> Dict[str, Any]:
    """Wrap legacy payload in the schema-based top-level envelope."""
    return {
        "schema_version": 1,
        "measurement_type": str(measurement_type),
        "beam_energy_keV": int(beam_energy_keV),
        "standards_by_mode": payload,
    }


def normalize_standards_file_payload(
    payload: Any,
    measurement_type: str,
    beam_energy_keV: int,
) -> Dict[str, Any]:
    """Normalize both legacy and schema-based standards payloads.

    Returns a schema-based payload that can be validated by the Pydantic models.
    """
    if not _is_mapping(payload):
        raise ValueError("Standards payload must be a dictionary")

    if is_legacy_standards_payload(payload):
        return migrate_legacy_standards_payload(
            payload=payload,
            measurement_type=measurement_type,
            beam_energy_keV=beam_energy_keV,
        )

    return dict(payload)
