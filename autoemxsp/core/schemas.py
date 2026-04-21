#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic schemas for ledger-based spectrum input handling.

Design:
- Shared acquisition attributes are stored once at sample-ledger level.
- Each spectrum entry stores only minimal per-spectrum fields.
- Fitter still consumes one resolved RawSpectralData at a time.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from autoemxsp.config.classes import (
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


class RawSpectralData(BaseModel):
    """Validated single-spectrum input for fitting.

    This model is intentionally minimal and is safe to construct independently
    in parallel quantifier instances.
    """

    spectrum_vals: List[float]  # counts per channel
    energy_vals: List[float]  # keV per channel
    collection_time: float  # seconds
    beam_energy: float  # keV
    emergence_angle: float  # degrees
    microscope_id: str
    meas_mode: str  # expected: "point" or "map"
    total_counts: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("spectrum_vals")
    @classmethod
    def validate_spectrum_vals(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Spectrum cannot be empty")
        if any(val < 0 for val in v):
            raise ValueError("Spectrum values cannot be negative")
        return v

    @field_validator("energy_vals")
    @classmethod
    def validate_energy_vals(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Energy axis cannot be empty")
        if any(v[i] >= v[i + 1] for i in range(len(v) - 1)):
            raise ValueError("Energy axis must be strictly increasing")
        return v

    @field_validator("collection_time")
    @classmethod
    def validate_collection_time(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Collection time must be positive")
        return v

    @field_validator("beam_energy")
    @classmethod
    def validate_beam_energy(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Beam energy must be positive")
        if v > 100:
            raise ValueError("Beam energy > 100 keV is unrealistic for EDS")
        return v

    @field_validator("emergence_angle")
    @classmethod
    def validate_emergence_angle(cls, v: float) -> float:
        if not 10 <= v <= 90:
            raise ValueError("Emergence angle must be between 10 and 90 degrees")
        return v

    @field_validator("meas_mode")
    @classmethod
    def validate_meas_mode(cls, v: str) -> str:
        valid = {"point", "map"}
        if v not in valid:
            raise ValueError(f"meas_mode must be one of {sorted(valid)}")
        return v

    @field_validator("total_counts")
    @classmethod
    def validate_total_counts(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("total_counts cannot be negative")
        return v

    @model_validator(mode="after")
    def validate_shape_and_total(self) -> "RawSpectralData":
        if len(self.spectrum_vals) != len(self.energy_vals):
            raise ValueError("spectrum_vals and energy_vals must have identical length")

        if self.total_counts is not None:
            computed = int(round(float(np.sum(self.spectrum_vals))))
            if self.total_counts != computed:
                raise ValueError(
                    f"total_counts ({self.total_counts}) does not match sum(spectrum_vals) ({computed})"
                )
        return self

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return spectrum and energy arrays for fitter consumption."""
        return np.asarray(self.spectrum_vals, dtype=float), np.asarray(self.energy_vals, dtype=float)


class SharedAcquisitionContext(BaseModel):
    """Sample-level fields shared by all spectra in one ledger."""

    energy_vals: List[float]  # keV per channel, identical for all spectra
    beam_energy: float  # keV
    emergence_angle: float  # degrees
    microscope_id: str
    meas_mode: str  # expected: "point" or "map"

    model_config = ConfigDict(extra="forbid")

    @field_validator("energy_vals")
    @classmethod
    def validate_energy_vals(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Energy axis cannot be empty")
        if any(v[i] >= v[i + 1] for i in range(len(v) - 1)):
            raise ValueError("Energy axis must be strictly increasing")
        return v

    @field_validator("beam_energy")
    @classmethod
    def validate_beam_energy(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Beam energy must be positive")
        if v > 100:
            raise ValueError("Beam energy > 100 keV is unrealistic for EDS")
        return v

    @field_validator("emergence_angle")
    @classmethod
    def validate_emergence_angle(cls, v: float) -> float:
        if not 10 <= v <= 90:
            raise ValueError("Emergence angle must be between 10 and 90 degrees")
        return v

    @field_validator("meas_mode")
    @classmethod
    def validate_meas_mode(cls, v: str) -> str:
        valid = {"point", "map"}
        if v not in valid:
            raise ValueError(f"meas_mode must be one of {sorted(valid)}")
        return v


class SpectrumEntry(BaseModel):
    """Minimal per-spectrum record inside a sample ledger."""

    spectrum_vals: List[float]  # counts per channel
    collection_time: float  # seconds
    total_counts: Optional[int] = None
    spectrum_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("spectrum_vals")
    @classmethod
    def validate_spectrum_vals(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("Spectrum cannot be empty")
        if any(val < 0 for val in v):
            raise ValueError("Spectrum values cannot be negative")
        return v

    @field_validator("collection_time")
    @classmethod
    def validate_collection_time(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Collection time must be positive")
        return v

    @field_validator("total_counts")
    @classmethod
    def validate_total_counts(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("total_counts cannot be negative")
        return v

    @model_validator(mode="after")
    def validate_total_if_present(self) -> "SpectrumEntry":
        if self.total_counts is not None:
            computed = int(round(float(np.sum(self.spectrum_vals))))
            if self.total_counts != computed:
                raise ValueError(
                    f"total_counts ({self.total_counts}) does not match sum(spectrum_vals) ({computed})"
                )
        return self


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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_and_cast_config_keys(cls, data: Any) -> Any:
        """Accept both *_cfg and *_config keys and cast dict payloads to dataclasses."""
        if not isinstance(data, dict):
            return data

        normalized: Dict[str, Any] = {}
        for key, value in data.items():
            if key.endswith("_config"):
                cfg_key = key[: -len("_config")] + "_cfg"
                normalized[cfg_key] = value
            else:
                normalized[key] = value

        config_types = {
            "microscope_cfg": MicroscopeConfig,
            "sample_cfg": SampleConfig,
            "measurement_cfg": MeasurementConfig,
            "sample_substrate_cfg": SampleSubstrateConfig,
            "quant_cfg": QuantConfig,
            "clustering_cfg": ClusteringConfig,
            "plot_cfg": PlotConfig,
            "powder_meas_cfg": PowderMeasurementConfig,
            "bulk_meas_cfg": BulkMeasurementConfig,
            "exp_stds_cfg": ExpStandardsConfig,
        }

        for key, cfg_type in config_types.items():
            value = normalized.get(key)
            if isinstance(value, dict):
                normalized[key] = cfg_type(**value)

        return normalized


class SampleLedger(BaseModel):
    """Single-sample ledger with shared context and per-spectrum entries."""

    ledger_id: str
    shared: SharedAcquisitionContext
    configs: Optional[LedgerConfigs] = None
    config_ref: Optional[str] = None
    spectra: List[SpectrumEntry]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("spectra")
    @classmethod
    def validate_non_empty_spectra(cls, v: List[SpectrumEntry]) -> List[SpectrumEntry]:
        if not v:
            raise ValueError("Ledger must contain at least one spectrum")
        return v

    @model_validator(mode="after")
    def validate_axis_lengths(self) -> "SampleLedger":
        n_channels = len(self.shared.energy_vals)
        for idx, spectrum in enumerate(self.spectra):
            if len(spectrum.spectrum_vals) != n_channels:
                raise ValueError(
                    f"Spectrum at index {idx} has {len(spectrum.spectrum_vals)} channels, "
                    f"expected {n_channels} from shared energy axis"
                )
                if self.configs is None and not self.config_ref:
                    raise ValueError("Provide either configs or config_ref in SampleLedger")
        return self

    def to_raw_spectrum(self, index: int) -> RawSpectralData:
        """Resolve one ledger entry into fitter-ready RawSpectralData."""
        s = self.spectra[index]
        return RawSpectralData(
            spectrum_vals=s.spectrum_vals,
            energy_vals=self.shared.energy_vals,
            total_counts=s.total_counts
            if s.total_counts is not None
            else int(round(float(np.sum(s.spectrum_vals)))),
            collection_time=s.collection_time,
            beam_energy=self.shared.beam_energy,
            emergence_angle=self.shared.emergence_angle,
            microscope_id=self.shared.microscope_id,
            meas_mode=self.shared.meas_mode,
        )

    def iter_raw_spectra(self) -> Iterator[RawSpectralData]:
        """Iterate fitter-ready spectra one at a time."""
        for i in range(len(self.spectra)):
            yield self.to_raw_spectrum(i)

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

        if payload.get("configs") is None:
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
