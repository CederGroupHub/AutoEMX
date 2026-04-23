#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic schemas for ledger-based spectrum input handling.

Design:
- Each spectrum entry is a pointer to an external .msa file stored under sample_path/spectra/.
- All acquisition context (energy axis, beam params, etc.) is captured in LedgerConfigs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from autoemx.config.classes import (
    BulkMeasurementConfig,
    ClusteringConfig,
    ExpStandardsConfig,
    MeasurementConfig,
    MicroscopeConfig,
    PlotConfig,
    PowderMeasurementConfig,
    QuantConfig,
    SampleConfig,
    SampleSubstrateConfig,
)


class Coordinate2D(BaseModel):
    """Simple 2D coordinate pair."""

    x: float
    y: float

    model_config = ConfigDict(extra="forbid")

    @field_validator("x", "y")
    @classmethod
    def validate_coordinate(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Coordinates must be finite")
        return v


class SpotCoordinates(BaseModel):
    """Acquisition spot coordinates in both machine and image spaces."""

    machine_coordinates: Optional[Coordinate2D] = None
    pixel_coordinates: Optional[Coordinate2D] = None

    model_config = ConfigDict(extra="forbid")


class AcquisitionDetails(BaseModel):
    """Structured acquisition context for one spectrum."""

    frame_id: Optional[str] = None
    particle_id: Optional[int] = None
    spot_coordinates: Optional[SpotCoordinates] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("frame_id")
    @classmethod
    def normalize_frame_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        normalized = v.strip()
        return normalized if normalized else None

    @field_validator("particle_id")
    @classmethod
    def validate_particle_id(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 0:
            raise ValueError("particle_id must be non-negative")
        return v


class SpectrumEntry(BaseModel):
    """Minimal per-spectrum record inside a sample ledger."""

    real_time: float  # seconds (wall-clock time)
    total_counts: Optional[int] = None
    spectrum_id: Optional[str] = None
    spectrum_relpath: Optional[str] = None  # relative to SampleLedger.sample_path (e.g., spectra/123.msa)
    instrument_background_relpath: Optional[str] = None  # relative to SampleLedger.sample_path (e.g., spectra/123_man_bckgrnd.npy)
    acquisition_details: Optional[AcquisitionDetails] = None
    metadata: Optional[Dict[str, Any]] = None
    quantification_results: List["QuantificationResult"] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def drop_legacy_fields(cls, data: Any) -> Any:
        """Drop removed legacy fields so old ledger JSON remains loadable."""
        if isinstance(data, dict):
            cleaned = dict(data)
            cleaned.pop("live_time", None)
            cleaned.pop("background_vals", None)
            cleaned.pop("spectrum_pointer_exists", None)
            cleaned.pop("original_spectrum_exists", None)
            cleaned.pop("spectrum_source", None)
            return cleaned
        return data

    @field_validator("real_time")
    @classmethod
    def validate_real_time(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Real time must be positive")
        return v

    @field_validator("total_counts")
    @classmethod
    def validate_total_counts(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("total_counts cannot be negative")
        return v

    @field_validator("spectrum_relpath")
    @classmethod
    def validate_spectrum_relpath(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        rel = v.strip()
        if not rel:
            return None
        path = Path(rel)
        if path.is_absolute():
            raise ValueError("spectrum_relpath must be relative to SampleLedger.sample_path")
        if ".." in path.parts:
            raise ValueError("spectrum_relpath cannot contain '..'")
        return rel

    @field_validator("instrument_background_relpath")
    @classmethod
    def validate_instrument_background_relpath(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        rel = v.strip()
        if not rel:
            return None
        path = Path(rel)
        if path.is_absolute():
            raise ValueError("instrument_background_relpath must be relative to SampleLedger.sample_path")
        if ".." in path.parts:
            raise ValueError("instrument_background_relpath cannot contain '..'")
        return rel

    @model_validator(mode="after")
    def validate_total_and_background(self) -> "SpectrumEntry":
        if self.spectrum_relpath is None:
            raise ValueError("spectrum_relpath is required for SpectrumEntry")

        return self


class FittedPeakResult(BaseModel):
    """Persisted fit summary for one fitted elemental line."""

    element: str
    line: str
    area: Optional[float] = None
    sigma: Optional[float] = None
    center: Optional[float] = None
    fwhm: Optional[float] = None
    peak_intensity: Optional[float] = None
    background_intensity: Optional[float] = None
    theoretical_energy: Optional[float] = None
    height: Optional[float] = None
    pb_ratio: Optional[float] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("element", "line")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("element and line cannot be empty")
        return v.strip()

    @field_validator(
        "area",
        "sigma",
        "center",
        "fwhm",
        "peak_intensity",
        "background_intensity",
        "theoretical_energy",
        "height",
        "pb_ratio",
    )
    @classmethod
    def validate_numeric_fields(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not np.isfinite(v):
            raise ValueError("Fit peak numeric values must be finite")
        return v


class FitResult(BaseModel):
    """Compact fit data persisted for downstream re-use without refitting."""

    r_squared: Optional[float] = None
    reduced_chi_squared: Optional[float] = None
    fitted_peaks: Dict[str, FittedPeakResult] = Field(default_factory=dict)
    reference_lines_by_element: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("r_squared")
    @classmethod
    def validate_r_squared(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not np.isfinite(v):
            raise ValueError("r_squared must be finite")
        return v

    @field_validator("reduced_chi_squared")
    @classmethod
    def validate_reduced_chi_squared(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (not np.isfinite(v) or v < 0):
            raise ValueError("reduced_chi_squared must be finite and non-negative")
        return v

    @field_validator("reference_lines_by_element")
    @classmethod
    def validate_reference_line_mapping(cls, v: Dict[str, str]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for element, el_line in v.items():
            if not element or not element.strip():
                raise ValueError("reference_lines_by_element contains an empty element key")
            if not el_line or not el_line.strip():
                raise ValueError("reference_lines_by_element contains an empty line reference")
            normalized[element.strip()] = el_line.strip()
        return normalized

    @model_validator(mode="after")
    def validate_reference_lines_exist(self) -> "FitResult":
        for element, el_line in self.reference_lines_by_element.items():
            if el_line not in self.fitted_peaks:
                raise ValueError(
                    f"reference line '{el_line}' for element '{element}' is not present in fitted_peaks"
                )

            peak = self.fitted_peaks[el_line]
            if peak.element != element:
                raise ValueError(
                    f"reference line '{el_line}' is mapped to element '{element}', "
                    f"but fitted peak stores element '{peak.element}'"
                )

        return self


class QuantificationDiagnostics(BaseModel):
    """Execution diagnostics captured during iterative quantification."""

    iterations_run: Optional[int] = None
    converged: Optional[bool] = None
    min_background_ref_lines: Optional[Dict[str, float]] = None
    missing_reference_peaks: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("iterations_run")
    @classmethod
    def validate_iterations_run(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("iterations_run must be non-negative")
        return v


class QuantificationConfig(BaseModel):
    """Configuration descriptor for one full quantification run."""

    quantification_id: str
    label: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("quantification_id")
    @classmethod
    def validate_quantification_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("quantification_id cannot be empty")
        return v.strip()


class QuantificationResult(BaseModel):
    """Persisted per-spectrum quantification output for a specific run/config."""

    quantification_id: str
    quant_flag: Optional[int] = None
    comment: Optional[str] = None
    composition_atomic_fractions: Optional[Dict[str, float]] = None
    composition_weight_fractions: Optional[Dict[str, float]] = None
    analytical_error: Optional[float] = None
    fit_result: Optional[FitResult] = None
    diagnostics: Optional[QuantificationDiagnostics] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("quantification_id")
    @classmethod
    def validate_non_empty_identifier(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("quantification_id cannot be empty")
        return v.strip()

    @field_validator("analytical_error")
    @classmethod
    def validate_analytical_error(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (not np.isfinite(v) or v < 0):
            raise ValueError("analytical_error must be finite and non-negative")
        return v


class LedgerConfigs(BaseModel):
    """Typed config bundle persisted in the ledger.

    Fields map one-to-one with existing runtime config classes.
    Optional configs depend on sample/measurement workflow.
    """

    microscope_cfg: MicroscopeConfig
    sample_cfg: SampleConfig
    measurement_cfg: MeasurementConfig
    sample_substrate_cfg: SampleSubstrateConfig
    quant_cfg: QuantConfig
    clustering_cfg: ClusteringConfig
    plot_cfg: PlotConfig
    powder_meas_cfg: Optional[PowderMeasurementConfig] = None
    bulk_meas_cfg: Optional[BulkMeasurementConfig] = None
    exp_stds_cfg: Optional[ExpStandardsConfig] = None

    model_config = ConfigDict(extra="forbid")


class SampleLedger(BaseModel):
    """Single-sample ledger with per-spectrum pointer entries and configuration bundle."""

    sample_id: str
    sample_path: str  # absolute base directory for relative pointers (e.g., spectra/*.msa)
    configs: LedgerConfigs
    spectra: List[SpectrumEntry]
    quantification_configs: List[QuantificationConfig] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("sample_path")
    @classmethod
    def validate_sample_path(cls, v: str) -> str:
        path = Path(v).expanduser()
        if not path.is_absolute():
            raise ValueError("sample_path must be an absolute path")
        return str(path)

    @model_validator(mode="after")
    def validate_ledger_integrity(self) -> "SampleLedger":
        config_ids = [cfg.quantification_id for cfg in self.quantification_configs]
        if len(config_ids) != len(set(config_ids)):
            raise ValueError("quantification_id values must be unique within a ledger")

        config_id_set = set(config_ids)
        for spectrum_index, spectrum in enumerate(self.spectra):
            if spectrum.spectrum_relpath is not None:
                resolved = (Path(self.sample_path) / spectrum.spectrum_relpath).resolve()
                sample_root = Path(self.sample_path).resolve()
                try:
                    resolved.relative_to(sample_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Spectrum at index {spectrum_index} has spectrum_relpath outside sample_path"
                    ) from exc

            if spectrum.instrument_background_relpath is not None:
                resolved = (Path(self.sample_path) / spectrum.instrument_background_relpath).resolve()
                sample_root = Path(self.sample_path).resolve()
                try:
                    resolved.relative_to(sample_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Spectrum at index {spectrum_index} has instrument_background_relpath outside sample_path"
                    ) from exc

            result_ids = set()
            for result in spectrum.quantification_results:
                if result.quantification_id not in config_id_set:
                    raise ValueError(
                        f"Spectrum at index {spectrum_index} contains a quantification result "
                        f"with unknown quantification_id '{result.quantification_id}'"
                    )

                if result.quantification_id in result_ids:
                    raise ValueError(
                        f"Spectrum at index {spectrum_index} contains duplicate "
                        f"quantification_id '{result.quantification_id}'"
                    )
                result_ids.add(result.quantification_id)

        return self

    @staticmethod
    def _load_counts_from_pointer_file(file_path: Path) -> List[float]:
        """Load counts from an external spectrum pointer file.

        Supported formats:
        - .msa / .msg: EMSA-like text file with optional headers and data section.
        - .json: object with a 'spectrum_vals' array.
        """
        suffix = file_path.suffix.lower()

        if suffix in {".msa", ".msg"}:
            counts: List[float] = []
            in_data_section = False
            with file_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("#SPECTRUM"):
                        in_data_section = True
                        continue
                    if line.startswith("#"):
                        continue
                    if not in_data_section:
                        continue

                    token = line.rstrip(",")
                    if "," in token:
                        token = token.split(",", maxsplit=1)[-1].strip()
                    try:
                        counts.append(float(token))
                    except ValueError:
                        continue

            if not counts:
                raise ValueError(f"No spectrum counts found in pointer file '{file_path}'")
            return counts

        if suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            values = payload.get("spectrum_vals")
            if not isinstance(values, list) or not values:
                raise ValueError(f"JSON pointer file '{file_path}' does not contain non-empty 'spectrum_vals'")
            return [float(v) for v in values]

        raise ValueError(f"Unsupported spectrum pointer extension '{suffix}' for file '{file_path}'")

    def append_quantification_config(self, quant_config: QuantificationConfig) -> None:
        """Append a quantification config, enforcing unique config identifiers."""
        if any(existing.quantification_id == quant_config.quantification_id for existing in self.quantification_configs):
            raise ValueError(
                f"quantification_id '{quant_config.quantification_id}' already exists in this ledger"
            )
        self.quantification_configs.append(quant_config)

    def append_quantification_result(self, spectrum_index: int, result: QuantificationResult) -> None:
        """Append a quantification result to the requested spectrum entry."""
        if spectrum_index < 0 or spectrum_index >= len(self.spectra):
            raise IndexError(f"spectrum_index {spectrum_index} is out of range")
        if result.quantification_id not in {cfg.quantification_id for cfg in self.quantification_configs}:
            raise ValueError(
                f"Quantification result references unknown quantification_id '{result.quantification_id}'"
            )
        if any(
            existing.quantification_id == result.quantification_id
            for existing in self.spectra[spectrum_index].quantification_results
        ):
            raise ValueError(
                f"Spectrum at index {spectrum_index} already contains quantification_id "
                f"'{result.quantification_id}'"
            )
        self.spectra[spectrum_index].quantification_results.append(result)

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> "SampleLedger":
        """Load and validate a ledger from JSON in one call."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.model_validate(payload)

    def to_dict(
        self,
        config_key_style: Literal["cfg", "config"] = "cfg",
        *,
        include_configs: bool = True,
    ) -> Dict[str, Any]:
        """Serialize ledger to a dict and choose config key suffix style.

        config_key_style:
        - "cfg": microscope_cfg, sample_cfg, ...
        - "config": microscope_config, sample_config, ...
        """
        payload = self.model_dump(mode="json")
        if not include_configs:
            payload.pop("configs", None)
            return payload

        if config_key_style == "cfg":
            return payload
        if config_key_style != "config":
            raise ValueError("config_key_style must be either 'cfg' or 'config'")

        cfg_block = payload.get("configs", {})
        renamed_cfg_block: Dict[str, Any] = {}
        for key, value in cfg_block.items():
            if key.endswith("_cfg"):
                renamed_cfg_block[key[: -len("_cfg")] + "_config"] = value
            else:
                renamed_cfg_block[key] = value
        payload["configs"] = renamed_cfg_block
        return payload

    def to_json_file(
        self,
        file_path: str | Path,
        *,
        config_key_style: Literal["cfg", "config"] = "cfg",
        include_configs: bool = True,
        indent: int = 2,
    ) -> None:
        """Write ledger to JSON in one call."""
        path = Path(file_path)
        payload = self.to_dict(config_key_style=config_key_style, include_configs=include_configs)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
            f.write("\n")


SpectrumEntry.model_rebuild()
