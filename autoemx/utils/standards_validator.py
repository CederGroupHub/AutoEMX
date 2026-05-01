#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal validation helpers for EDS standards JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

_KEY_ID = "ID"
_KEY_DATETIME = "datetime"
_KEY_CORRECTED_PB = "Corrected_PB"
_KEY_STDEV_PB = "Stdev_PB"
_KEY_REL_STDEV_PB = "Rel_stdev_PB (%)"
_KEY_FORMULA = "Formula"
_KEY_STD_TYPE = "Std_type"
_KEY_MEASURED_PB = "Measured_PB"
_KEY_USE_FOR_MEAN = "Use_for_mean_calc"
_KEY_MEAN_Z = "Mean_Z"

_KEY_Z_MASS = "mass-averaged"
_KEY_Z_ATOMIC = "atomic-averaged"
_KEY_Z_STATHAM = "Statham2016"
_KEY_Z_MARKOWICZ = "Markowicz1984"

_STANDARDS_FILENAME_RE = re.compile(r"^(?P<meas_type>.+)_Stds_(?P<beam_energy>\d+)keV\.json$")


class _StandardMeanZ(BaseModel):
    mass_averaged: float = Field(alias=_KEY_Z_MASS)
    atomic_averaged: float = Field(alias=_KEY_Z_ATOMIC)
    statham2016: float = Field(alias=_KEY_Z_STATHAM)
    markowicz1984: float = Field(alias=_KEY_Z_MARKOWICZ)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class _StandardEntry(BaseModel):
    standard_id: str = Field(alias=_KEY_ID)
    datetime: str = Field(alias=_KEY_DATETIME)
    corrected_pb: float = Field(alias=_KEY_CORRECTED_PB)
    stdev_pb: float = Field(alias=_KEY_STDEV_PB)
    rel_stdev_pb_percent: float = Field(alias=_KEY_REL_STDEV_PB)
    formula: Optional[str] = Field(default=None, alias=_KEY_FORMULA)
    std_type: Optional[str] = Field(default=None, alias=_KEY_STD_TYPE)
    measured_pb: Optional[float] = Field(default=None, alias=_KEY_MEASURED_PB)
    use_for_mean_calc: Optional[bool] = Field(default=None, alias=_KEY_USE_FOR_MEAN)
    mean_z: Optional[_StandardMeanZ] = Field(default=None, alias=_KEY_MEAN_Z)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("standard_id", "datetime")
    @classmethod
    def _validate_non_empty_strings(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Standard entry identifiers and timestamps cannot be empty")
        return normalized


class _EDSStandardsFile(BaseModel):
    schema_version: int = 1
    measurement_type: str
    beam_energy_keV: int
    standards_by_mode: Dict[str, Dict[str, List[_StandardEntry]]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("schema_version must be >= 1")
        return value

    @field_validator("measurement_type")
    @classmethod
    def _validate_measurement_type(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("measurement_type cannot be empty")
        return normalized

    @field_validator("beam_energy_keV", mode="before")
    @classmethod
    def _validate_beam_energy(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("beam_energy_keV must be an integer")
        normalized = int(value)
        if normalized <= 0:
            raise ValueError("beam_energy_keV must be positive")
        return normalized


def normalize_standards_payload(payload: Any, meas_type: str, beam_energy: int) -> Dict[str, Any]:
    """Normalize standards payload to the schema envelope.

    Legacy payload shape with top-level modes is automatically wrapped.
    """
    if not isinstance(payload, dict):
        raise ValueError("Standards payload must be a dictionary")

    if "standards_by_mode" in payload:
        return dict(payload)

    return {
        "schema_version": 1,
        "measurement_type": str(meas_type),
        "beam_energy_keV": int(beam_energy),
        "standards_by_mode": payload,
    }


def iter_standards_files(root_dir: Path) -> Iterable[Path]:
    """Yield candidate standards files under root directory."""
    yield from sorted(root_dir.rglob("*_Stds_*keV.json"))


def parse_filename_metadata(file_path: Path) -> Tuple[str, int]:
    """Extract measurement type and beam energy from standards filename."""
    match = _STANDARDS_FILENAME_RE.match(file_path.name)
    if match is None:
        raise ValueError(
            f"Filename '{file_path.name}' does not match '<MeasurementType>_Stds_<keV>keV.json'"
        )
    meas_type = match.group("meas_type")
    beam_energy = int(match.group("beam_energy"))
    return meas_type, beam_energy


def validate_standards_file(file_path: Path) -> Tuple[bool, str]:
    """Validate one standards JSON file and return status plus detail string."""
    try:
        meas_type, beam_energy = parse_filename_metadata(file_path)
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        normalized_payload = normalize_standards_payload(payload, meas_type=meas_type, beam_energy=beam_energy)
        _EDSStandardsFile.model_validate(normalized_payload)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def validate_standards_tree(root_dir: Path, verbose: bool = True) -> int:
    """Validate all standards files under root directory and return process-style exit code."""
    files = list(iter_standards_files(root_dir))
    if not files:
        if verbose:
            print(f"No standards files found under: {root_dir}")
        return 0

    invalid_files: List[Tuple[Path, str]] = []
    if verbose:
        print(f"Found {len(files)} standards file(s) under: {root_dir}")

    for standards_file in files:
        is_valid, details = validate_standards_file(standards_file)
        relative_path = standards_file.as_posix()
        if is_valid:
            if verbose:
                print(f"[OK]   {relative_path}")
        else:
            if verbose:
                print(f"[FAIL] {relative_path}")
                print(f"       {details}")
            invalid_files.append((standards_file, details))

    if invalid_files:
        if verbose:
            print(f"\nValidation failed for {len(invalid_files)} file(s).")
        return 1

    if verbose:
        print("\nAll standards files are schema-valid.")
    return 0
