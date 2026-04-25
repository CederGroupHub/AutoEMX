#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .fitting import FitResult


class QuantificationDiagnostics(BaseModel):
    """
    Execution diagnostics captured during iterative quantification.

    Attributes
    ----------
    iterations_run : int, optional
        Number of quantification iterations performed.
    converged : bool, optional
        True if the iterative fit converged within the allowed iterations.
    interrupted : bool, optional
        True when no composition result was produced, either because the spectrum
        failed a pre-fit validity check (insufficient counts, background too low,
        etc.) or because the iterative fit was aborted early after detecting signs
        of an unreliable result (poor fit quality, excessive analytical error, or
        excessive X-ray absorption) while ``interrupt_fits_bad_spectra=True``.
        Spectra with ``interrupted=True`` are automatically re-quantified on the
        next run when ``interrupt_fits_bad_spectra=False``.
    min_background_ref_lines : float, optional
        Minimum background counts under any reference peak used for quantification.
    missing_reference_peaks : list of str, optional
        Reference peaks that were absent or below the minimum acceptable PB ratio.
    """

    iterations_run: Optional[int] = None
    converged: Optional[bool] = None
    interrupted: Optional[bool] = None
    min_background_ref_lines: Optional[float] = None
    missing_reference_peaks: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("iterations_run")
    @classmethod
    def validate_iterations_run(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("iterations_run must be non-negative")
        return v

    @field_validator("min_background_ref_lines")
    @classmethod
    def validate_min_background_ref_lines(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return None
        normalized_value = float(value)
        if not np.isfinite(normalized_value) or normalized_value < 0:
            raise ValueError("min_background_ref_lines must be finite and non-negative")
        return normalized_value


def _validate_quantification_id(value: Any) -> Any:
    """Reject bool ids explicitly and let Pydantic coerce normal integer values."""
    if isinstance(value, bool):
        raise ValueError("quantification_id must be a non-negative integer")
    return value


def _canonicalize_json_value(value: Any) -> Any:
    """Recursively normalize values to deterministic JSON-compatible primitives."""
    if isinstance(value, dict):
        return {
            str(k): _canonicalize_json_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_payload_differences(
    *,
    left: Any,
    right: Any,
    path: str,
    out: Dict[str, Dict[str, Any]],
) -> None:
    """Recursively collect payload differences keyed by a dotted path."""
    if isinstance(left, dict) and isinstance(right, dict):
        all_keys = sorted(set(left) | set(right))
        for key in all_keys:
            next_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                out[next_path] = {"old": None, "new": right[key]}
                continue
            if key not in right:
                out[next_path] = {"old": left[key], "new": None}
                continue
            _collect_payload_differences(
                left=left[key],
                right=right[key],
                path=next_path,
                out=out,
            )
        return

    if isinstance(left, list) and isinstance(right, list):
        if left != right:
            out[path or "root"] = {"old": left, "new": right}
        return

    if left != right:
        out[path or "root"] = {"old": left, "new": right}


class ClusteringConfig(BaseModel):
    """Configuration for clustering of compositions and their filtering.

    Attributes:
        method (str): Clustering algorithm to use. Allowed: 'kmeans' (implemented), 'dbscan' (not implemented).
        features (str): Feature set to use for clustering.
        k_forced (Optional[int]): Forced number of clusters when `k_finding_method` is "forced".
        k_resolved (Optional[int]): Number of clusters resolved from data for this run.
        k_finding_method (str): Method to determine the number of clusters. Set to "forced" when `k_forced` is specified manually.
            Allowed methods are "silhouette", "calinski_harabasz", "elbow".
        max_k (int): Maximum allowed number of clusters.
        ref_formulae (List[str]): List of possible phases present in the sample, as chemical formula strings.
        do_matrix_decomposition (bool) : Whether to compute matrix decomposition for intermixed phases. Slow if many candidate phases are provided. Default: True
        max_analytical_error_percent (Optional[float]): Maximum analytical error acceptable for composition to be considered in phase determination, expressed as w%. Can be None.
        min_bckgrnd_cnts (Optional[float]): Minimum background counts required under reference lines for quantification to be accepted.
            Set to None to disable this threshold-based filter.
        quant_flags_accepted (List[int]): List of quantification flags considered acceptable, others are filtered out prior clustering.
            Quantification flags indicate whether the quantification or the fit of each spectrum is likely to be affected by large errors:
               - 0: Quantification is ok, although it may be affected by large analytical error
               - -1: As above, but quantification did not converge within 30 steps
               - 1: Error during EDS acquisition. No fit executed
               - 2: Total number of counts is lower than 95% of target counts, likely due to wrong segmentation. No fit executed
               - 3: Spectrum has too low signal in its low-energy portion, leading to poor quantification in this region. No fit executed
               - 4: Poor fit. Fit interrupted if interrupt_fits_bad_spectra=True
               - 5: Too high analytical error (>50%) indicating a missing element or other major sources of error. Fit interrupted if interrupt_fits_bad_spectra=True
               - 6: Excessive X-ray absorption. Fit interrupted if interrupt_fits_bad_spectra=True
               - 7: Excessive signal contamination from substrate
               - 8: Too few background counts below reference peak, likely leading to large quantification errors
               - 9: Unknown fitting error
    """

    clustering_id: int = 0
    method: str = "kmeans"
    features: str = "at_fr"
    k_forced: Optional[int] = None
    k_resolved: Optional[int] = None
    k_finding_method: str = "silhouette"
    max_k: int = 6
    ref_formulae: List[str] = Field(default_factory=list)
    do_matrix_decomposition: bool = True
    max_analytical_error_percent: Optional[float] = 5.0
    min_bckgrnd_cnts: Optional[float] = 5.0
    quant_flags_accepted: List[int] = Field(default_factory=lambda: [0, -1])

    model_config = ConfigDict(extra="forbid")

    @field_validator("clustering_id", mode="before")
    @classmethod
    def validate_clustering_id_input(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("clustering_id must be a non-negative integer")
        return v

    @field_validator("clustering_id")
    @classmethod
    def validate_clustering_id(cls, value: int) -> int:
        if value < 0:
            raise ValueError("clustering_id must be non-negative")
        return value

    @field_validator("method", "features", "k_finding_method")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Clustering string fields cannot be empty")
        return normalized

    @field_validator("max_k")
    @classmethod
    def validate_max_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_k must be positive")
        return value

    @field_validator("k_forced", "k_resolved")
    @classmethod
    def validate_k(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("k values must be positive when provided")
        return value

    @model_validator(mode="after")
    def validate_k_semantics(self) -> "ClusteringConfig":
        """Enforce forced-k semantics and normalize method when a forced k is provided."""
        forced_key = "forced"
        if self.k_forced is not None and self.k_finding_method != forced_key:
            self.k_finding_method = forced_key
        if self.k_forced is None and self.k_finding_method == forced_key:
            raise ValueError("k_finding_method='forced' requires k_forced to be set")
        return self

    @field_validator("max_analytical_error_percent")
    @classmethod
    def validate_max_analytical_error_percent(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not np.isfinite(value):
            raise ValueError("max_analytical_error_percent must be finite when provided")
        return value

    @field_validator("min_bckgrnd_cnts")
    @classmethod
    def validate_min_bckgrnd_cnts(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("min_bckgrnd_cnts must be non-negative or None")
        return value

    @field_validator("ref_formulae")
    @classmethod
    def validate_ref_formulae(cls, values: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in values:
            formula = str(value).strip()
            if not formula:
                continue
            if formula not in seen:
                normalized.append(formula)
                seen.add(formula)
        return normalized

    @field_validator("quant_flags_accepted")
    @classmethod
    def validate_quant_flags_accepted(cls, values: List[int]) -> List[int]:
        normalized: List[int] = []
        seen = set()
        for value in values:
            numeric = int(value)
            if numeric not in seen:
                normalized.append(numeric)
                seen.add(numeric)
        return normalized

    def fingerprint_payload(self) -> Dict[str, Any]:
        """Return canonical clustering inputs used to decide clustering reuse."""
        return {
            "method": self.method,
            "features": self.features,
            "k_forced": self.k_forced,
            "k_finding_method": self.k_finding_method,
            "max_k": self.max_k,
            "ref_formulae": sorted(self.ref_formulae),
            "do_matrix_decomposition": self.do_matrix_decomposition,
            "max_analytical_error_percent": self.max_analytical_error_percent,
            "min_bckgrnd_cnts": self.min_bckgrnd_cnts,
            "quant_flags_accepted": sorted(self.quant_flags_accepted),
        }

    def fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint for clustering scientific inputs."""
        canonical_payload = _canonicalize_json_value(self.fingerprint_payload())
        canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def fingerprint_differences(self, other: "ClusteringConfig") -> Dict[str, Dict[str, Any]]:
        """Return a deterministic diff of clustering scientific inputs against another config."""
        left_payload = _canonicalize_json_value(self.fingerprint_payload())
        right_payload = _canonicalize_json_value(other.fingerprint_payload())

        differences: Dict[str, Dict[str, Any]] = {}
        _collect_payload_differences(
            left=left_payload,
            right=right_payload,
            path="",
            out=differences,
        )
        return dict(sorted(differences.items()))


class QuantificationConfig(BaseModel):
    """Configuration descriptor for one full quantification run.

    The `options` mapping stores quantification-method options used to decide
    whether a previous run can be resumed (for example
    `use_project_specific_std_dict`).
    
    The `reference_lines_by_element` mapping records which spectral line was selected
    for each element during this quantification run. This enables comparison between
    runs to detect whether line selection changed, which would require requantification.
    """

    quantification_id: int
    label: Optional[str] = None
    sample_elements: List[str] = Field(default_factory=list)
    substrate_elements: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)
    reference_values_by_el_line: Dict[str, Any] = Field(default_factory=dict)
    reference_lines_by_element: Dict[str, str] = Field(default_factory=dict)
    active_clustering_cfg_index: Optional[int] = None
    clustering_configs: List[ClusteringConfig] = Field(default_factory=list)


    model_config = ConfigDict(extra="forbid")

    @field_validator("quantification_id", mode="before")
    @classmethod
    def validate_quantification_id_input(cls, v: Any) -> Any:
        return _validate_quantification_id(v)

    @field_validator("quantification_id")
    @classmethod
    def validate_quantification_id(cls, v: int) -> int:
        if v < 0:
            raise ValueError("quantification_id must be non-negative")
        return v

    @field_validator("sample_elements", "substrate_elements")
    @classmethod
    def validate_element_lists(cls, values: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in values:
            element = value.strip()
            if not element:
                raise ValueError("Element lists cannot contain empty entries")
            if element not in seen:
                normalized.append(element)
                seen.add(element)
        return normalized

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = {"method", "spectrum_lims", "fit_tolerance", "use_instrument_background"}
        missing_keys = required_keys - set(value)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"options is missing required quantification keys: {missing}")

        spectrum_lims = value.get("spectrum_lims")
        if not isinstance(spectrum_lims, (list, tuple)) or len(spectrum_lims) != 2:
            raise ValueError("options['spectrum_lims'] must contain exactly two values")

        low_raw = float(spectrum_lims[0])
        high_raw = float(spectrum_lims[1])
        if not np.isfinite(low_raw) or not np.isfinite(high_raw):
            raise ValueError("options['spectrum_lims'] values must be finite")
        if not low_raw.is_integer() or not high_raw.is_integer():
            raise ValueError("options['spectrum_lims'] values must be integer channel indices")

        low = int(low_raw)
        high = int(high_raw)
        if low >= high:
            raise ValueError("options['spectrum_lims'] must satisfy low < high")

        normalized = dict(value)
        normalized["spectrum_lims"] = [low, high]
        normalized["fit_tolerance"] = float(normalized["fit_tolerance"])
        normalized["use_instrument_background"] = bool(normalized["use_instrument_background"])
        if "use_project_specific_std_dict" in normalized:
            normalized["use_project_specific_std_dict"] = bool(normalized["use_project_specific_std_dict"])
        normalized["method"] = str(normalized["method"]).strip()
        if not normalized["method"]:
            raise ValueError("options['method'] cannot be empty")
        return normalized

    @field_validator("reference_values_by_el_line")
    @classmethod
    def validate_reference_values_by_el_line(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for el_line, ref_value in value.items():
            key = str(el_line).strip()
            if not key:
                raise ValueError("reference_values_by_el_line cannot contain empty keys")
            normalized[key] = ref_value
        return normalized

    @field_validator("reference_lines_by_element")
    @classmethod
    def validate_reference_lines_by_element(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that element→line mapping contains no empty keys or values."""
        normalized: Dict[str, str] = {}
        for element, el_line in v.items():
            if not element or not element.strip():
                raise ValueError("reference_lines_by_element contains an empty element key")
            if not el_line or not el_line.strip():
                raise ValueError("reference_lines_by_element contains an empty line reference")
            normalized[element.strip()] = el_line.strip()
        return normalized

    @model_validator(mode="after")
    def validate_clustering_config_index(self) -> "QuantificationConfig":
        """Ensure active clustering config index is valid and defaults to the last config."""

        if self.clustering_configs:
            if self.active_clustering_cfg_index is None:
                self.active_clustering_cfg_index = len(self.clustering_configs) - 1
            elif not (0 <= self.active_clustering_cfg_index < len(self.clustering_configs)):
                raise ValueError("active_clustering_cfg_index must reference an existing clustering config")
        elif self.active_clustering_cfg_index is not None:
            raise ValueError("active_clustering_cfg_index must be None when no clustering configs are available")
        return self

    def get_active_clustering_config(self) -> Optional[ClusteringConfig]:
        """Return the active clustering config, defaulting to the last one when available."""
        if not self.clustering_configs:
            return None
        if self.active_clustering_cfg_index is None:
            return self.clustering_configs[-1]
        return self.clustering_configs[self.active_clustering_cfg_index]

    @staticmethod
    def derive_reference_lines_by_element(
        reference_values_by_el_line: Dict[str, Any],
        preferred_lines: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        """Build a deterministic element->line map from available reference values."""
        priority: Dict[str, int] = {
            str(line): idx for idx, line in enumerate(preferred_lines or [])
        }

        selected: Dict[str, str] = {}
        for raw_key in sorted(reference_values_by_el_line):
            key = str(raw_key).strip()
            if "_" not in key:
                continue

            element, line = key.split("_", 1)
            element = element.strip()
            line = line.strip()
            if not element or not line:
                continue

            existing = selected.get(element)
            if existing is None:
                selected[element] = key
                continue

            existing_line = existing.split("_", 1)[1]
            existing_rank = priority.get(existing_line, len(priority))
            candidate_rank = priority.get(line, len(priority))

            if candidate_rank < existing_rank or (
                candidate_rank == existing_rank and key < existing
            ):
                selected[element] = key

        return dict(sorted(selected.items()))

    def fingerprint_payload(self) -> Dict[str, Any]:
        """Return canonical scientific inputs used to decide quantification reuse."""
        return {
            "sample_elements": sorted(self.sample_elements),
            "substrate_elements": sorted(self.substrate_elements),
            "options": _canonicalize_json_value(self.options),
            "reference_values_by_el_line": _canonicalize_json_value(self.reference_values_by_el_line),
            "reference_lines_by_element": _canonicalize_json_value(self.reference_lines_by_element),
        }

    def fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint for quantification scientific inputs."""
        canonical_payload = self.fingerprint_payload()
        canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def fingerprint_differences(self, other: "QuantificationConfig") -> Dict[str, Dict[str, Any]]:
        """Return a deterministic diff of scientific inputs against another config."""
        left_payload = self.fingerprint_payload()
        right_payload = other.fingerprint_payload()

        differences: Dict[str, Dict[str, Any]] = {}
        self._collect_payload_differences(
            left=left_payload,
            right=right_payload,
            path="",
            out=differences,
        )
        return dict(sorted(differences.items()))

    @staticmethod
    def _canonicalize_json_value(value: Any) -> Any:
        """Backward-compatible wrapper around module-level canonicalization."""
        return _canonicalize_json_value(value)

    @staticmethod
    def _collect_payload_differences(
        *,
        left: Any,
        right: Any,
        path: str,
        out: Dict[str, Dict[str, Any]],
    ) -> None:
        """Backward-compatible wrapper around module-level payload diffing."""
        _collect_payload_differences(left=left, right=right, path=path, out=out)


class QuantificationResult(BaseModel):
    """Persisted per-spectrum quantification output for a specific run/config."""

    quantification_id: int
    quant_flag: Optional[int] = None
    comment: Optional[str] = None
    composition_atomic_fractions: Optional[Dict[str, float]] = None
    composition_weight_fractions: Optional[Dict[str, float]] = None
    analytical_error: Optional[float] = None
    fit_result: Optional[FitResult] = None
    diagnostics: Optional[QuantificationDiagnostics] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("quantification_id", mode="before")
    @classmethod
    def validate_quantification_id_input(cls, v: Any) -> Any:
        return _validate_quantification_id(v)

    @field_validator("quantification_id")
    @classmethod
    def validate_non_empty_identifier(cls, v: int) -> int:
        if v < 0:
            raise ValueError("quantification_id must be non-negative")
        return v

    @field_validator("analytical_error")
    @classmethod
    def validate_analytical_error(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not np.isfinite(v):
            raise ValueError("analytical_error must be finite")
        return v
