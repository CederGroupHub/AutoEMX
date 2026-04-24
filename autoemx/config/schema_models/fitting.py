#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
