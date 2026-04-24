#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy Data.csv loaders used while bootstrapping ledger creation."""

import importlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import autoemx.utils.constants as cnst
from autoemx.config.schemas import (
    AcquisitionDetails,
    ClusteringConfig,
    Coordinate2D,
    FitResult,
    QuantificationConfig,
    QuantificationDiagnostics,
    QuantificationResult,
    SpotCoordinates,
)

_MIN_BACKGROUND_COMMENT_PATTERN = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s+min\.\s+ref\.\s+bckgrnd\s+counts",
    re.IGNORECASE,
)


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value) or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value) or value == "":
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _resolve_first_present_column(data_df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first existing column from candidates using case-insensitive fallback."""
    if data_df is None or data_df.empty:
        return None

    for column in candidates:
        if column in data_df.columns:
            return column

    normalized = {str(col).strip().lower(): col for col in data_df.columns}
    for column in candidates:
        key = str(column).strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def _parse_analytical_error_percent(value: Any) -> Optional[float]:
    """Parse legacy `An Er w%` values into a signed fraction."""
    analytical_error_percent = _parse_optional_float(value)
    if analytical_error_percent is None:
        return None
    return analytical_error_percent / 100.0


def _coerce_pixel_coordinate(value: Any) -> Optional[int]:
    """Round finite pixel coordinates to integer image-space values."""
    numeric = _parse_optional_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _extract_reference_values_for_quantification(
    standards_by_line: Dict[str, Any],
    relevant_elements: set[str],
) -> Dict[str, Any]:
    """Mirror live quantification by keeping only quantifier reference-line families."""
    from autoemx.core.quantifier.quantifier import XSp_Quantifier

    reference_line_suffixes = tuple(XSp_Quantifier.xray_quant_ref_lines)
    reference_values_by_el_line: Dict[str, Any] = {}

    for el_line, std_values in standards_by_line.items():
        element, _, line_suffix = str(el_line).partition("_")
        if element not in relevant_elements or line_suffix not in reference_line_suffixes:
            continue
        mean_std = next(
            (std for std in std_values if std.get(cnst.STD_ID_KEY) == cnst.STD_MEAN_ID_KEY),
            None,
        )
        if mean_std is None or cnst.COR_PB_DF_KEY not in mean_std:
            continue
        reference_values_by_el_line[str(el_line)] = float(mean_std[cnst.COR_PB_DF_KEY])

    return dict(sorted(reference_values_by_el_line.items()))


def _extract_min_background_ref_lines_from_comment(comment: Optional[str]) -> Optional[float]:
    """Extract the legacy minimum reference-background counts summary from comments when present."""
    if not comment:
        return None
    match = _MIN_BACKGROUND_COMMENT_PATTERN.search(comment)
    if match is None:
        return None
    min_background_counts = _parse_optional_float(match.group(1))
    if min_background_counts is None:
        return None
    return min_background_counts


def _load_legacy_quant_cfg(sample_result_dir: str) -> Optional[Any]:
    """Load the raw legacy quantification config when LedgerConfigs has already dropped it."""
    from autoemx.config.classes import QuantificationOptionsConfig

    candidate_files = [
        Path(sample_result_dir) / f"{cnst.CONFIG_FILENAME}.json",
        Path(sample_result_dir) / f"{cnst.ACQUISITION_INFO_FILENAME}.json",
    ]
    for config_path in candidate_files:
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_quant_cfg = payload.get(cnst.QUANTIFICATION_CFG_KEY)
        if raw_quant_cfg is None:
            continue
        try:
            return QuantificationOptionsConfig.model_validate(raw_quant_cfg)
        except Exception:
            continue
    return None


def _load_legacy_clustering_cfg(sample_result_dir: str) -> Optional[ClusteringConfig]:
    """Load the raw legacy clustering config from legacy JSON files."""
    candidate_files = [
        Path(sample_result_dir) / f"{cnst.CONFIG_FILENAME}.json",
        Path(sample_result_dir) / f"{cnst.ACQUISITION_INFO_FILENAME}.json",
    ]
    for config_path in candidate_files:
        if not config_path.exists():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_clustering_cfg = payload.get(cnst.CLUSTERING_CFG_KEY)
        if raw_clustering_cfg is None:
            continue
        try:
            normalized_payload = dict(raw_clustering_cfg)

            # Legacy configs used `k`; map it to the new `k_forced` field.
            if "k_forced" not in normalized_payload and "k" in normalized_payload:
                normalized_payload["k_forced"] = normalized_payload.get("k")
            normalized_payload.pop("k", None)

            # Keep only fields declared in the schema to avoid extra-field validation errors.
            allowed_fields = set(ClusteringConfig.model_fields.keys())
            normalized_payload = {
                key: value
                for key, value in normalized_payload.items()
                if key in allowed_fields
            }

            return ClusteringConfig.model_validate(normalized_payload)
        except Exception:
            continue
    return None


def _load_em_driver(microscope_id: Optional[str]):
    """Import the microscope driver module from microscope ID when available."""
    if microscope_id is None:
        return None
    module_name = str(microscope_id).strip()
    if not module_name:
        return None
    try:
        return importlib.import_module(f"autoemx.microscope_drivers.{module_name}")
    except Exception:
        return None


def _resolve_frame_dimensions(
    sample_result_dir: str,
    frame_id: Optional[str],
    em_driver,
    cache: Dict[str, Optional[Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    """Resolve frame dimensions from saved images, then fallback to driver defaults."""
    cache_key = str(frame_id).strip() if frame_id is not None else "__default__"
    if cache_key in cache:
        return cache[cache_key]

    images_dir = Path(sample_result_dir, cnst.IMAGES_DIR)
    allowed_suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

    if images_dir.exists():
        candidate_paths: List[Path] = []
        frame_token = None
        if frame_id is not None:
            frame_id_str = str(frame_id).strip()
            if frame_id_str:
                frame_token = f"_fr{frame_id_str}"

        if frame_token is not None:
            for path in images_dir.iterdir():
                if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                    continue
                if frame_token in path.stem:
                    candidate_paths.append(path)

        if not candidate_paths:
            candidate_paths = [
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in allowed_suffixes
            ]

        for image_path in sorted(candidate_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None or len(getattr(image, "shape", ())) < 2:
                continue
            dims = (int(image.shape[1]), int(image.shape[0]))
            cache[cache_key] = dims
            cache.setdefault("__default__", dims)
            return dims

    if em_driver is not None:
        driver_w = _parse_optional_int(getattr(em_driver, "im_width", None))
        driver_h = _parse_optional_int(getattr(em_driver, "im_height", None))
        if driver_w is not None and driver_h is not None and driver_w > 0 and driver_h > 0:
            dims = (driver_w, driver_h)
            cache[cache_key] = dims
            cache.setdefault("__default__", dims)
            return dims

    cache[cache_key] = None
    return None


def convert_machine_to_pixel_coordinates(
    machine_x: Any,
    machine_y: Any,
    *,
    sample_result_dir: str,
    frame_id: Optional[str],
    microscope_id: Optional[str],
    cache: Optional[Dict[str, Optional[Tuple[int, int]]]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Convert normalized machine coordinates to pixel coordinates with em_driver."""
    x_machine = _parse_optional_float(machine_x)
    y_machine = _parse_optional_float(machine_y)
    if x_machine is None or y_machine is None:
        return None, None

    local_cache: Dict[str, Optional[Tuple[int, int]]] = cache if cache is not None else {}
    em_driver = _load_em_driver(microscope_id)
    dims = _resolve_frame_dimensions(sample_result_dir, frame_id, em_driver, local_cache)
    if dims is None:
        return None, None

    image_width, image_height = dims
    if image_width <= 0 or image_height <= 0:
        return None, None

    if em_driver is not None and hasattr(em_driver, "frame_rel_to_pixel_coords"):
        try:
            pixel_coords = em_driver.frame_rel_to_pixel_coords(
                np.asarray([[x_machine, y_machine]], dtype=float),
                int(image_width),
                int(image_height),
            )
            if pixel_coords is not None:
                x_pixel = _parse_optional_float(pixel_coords[0][0])
                y_pixel = _parse_optional_float(pixel_coords[0][1])
                return x_pixel, y_pixel
        except Exception:
            pass

    # Fallback to same transform if driver function is unavailable.
    aspect_ratio = float(image_height) / float(image_width)
    if aspect_ratio == 0:
        return None, None
    x_pixel = (x_machine + 0.5) * float(image_width)
    y_pixel = (y_machine / aspect_ratio + 0.5) * float(image_height)
    return x_pixel, y_pixel


def load_legacy_acquisition_details_by_spectrum_id(
    data_csv_path: str,
    *,
    sample_result_dir: str,
    microscope_id: Optional[str],
) -> Dict[str, AcquisitionDetails]:
    """Load acquisition details keyed by spectrum id from legacy Data.csv."""
    details_by_id: Dict[str, AcquisitionDetails] = {}
    if not data_csv_path or not os.path.exists(data_csv_path):
        return details_by_id

    try:
        data_df = pd.read_csv(data_csv_path)
    except Exception:
        return details_by_id

    dims_cache: Dict[str, Optional[Tuple[int, int]]] = {}

    for row_idx, row in data_df.iterrows():
        spectrum_id_val = row.get(cnst.SP_ID_DF_KEY, row_idx)
        if isinstance(spectrum_id_val, (int, float, np.integer)) and not pd.isna(spectrum_id_val):
            spectrum_id = str(int(spectrum_id_val))
        else:
            spectrum_id = str(spectrum_id_val)
        if not spectrum_id:
            spectrum_id = str(row_idx)

        frame_id = (
            str(row.get(cnst.FRAME_ID_DF_KEY)).strip()
            if row.get(cnst.FRAME_ID_DF_KEY) is not None and not pd.isna(row.get(cnst.FRAME_ID_DF_KEY))
            else None
        )

        machine_x = row.get(cnst.SP_X_COORD_DF_KEY)
        machine_y = row.get(cnst.SP_Y_COORD_DF_KEY)
        pixel_x = _coerce_pixel_coordinate(row.get(cnst.SP_X_PIXEL_COORD_DF_KEY))
        pixel_y = _coerce_pixel_coordinate(row.get(cnst.SP_Y_PIXEL_COORD_DF_KEY))
        if pixel_x is None or pixel_y is None:
            conv_x, conv_y = convert_machine_to_pixel_coordinates(
                machine_x,
                machine_y,
                sample_result_dir=sample_result_dir,
                frame_id=frame_id,
                microscope_id=microscope_id,
                cache=dims_cache,
            )
            if pixel_x is None:
                pixel_x = _coerce_pixel_coordinate(conv_x)
            if pixel_y is None:
                pixel_y = _coerce_pixel_coordinate(conv_y)

        machine_coordinates = None
        x_machine = _parse_optional_float(machine_x)
        y_machine = _parse_optional_float(machine_y)
        if x_machine is not None and y_machine is not None:
            machine_coordinates = Coordinate2D(x=x_machine, y=y_machine)

        pixel_coordinates = None
        if pixel_x is not None and pixel_y is not None:
            pixel_coordinates = (pixel_x, pixel_y)

        spot_coordinates = None
        if machine_coordinates is not None or pixel_coordinates is not None:
            spot_coordinates = SpotCoordinates(
                machine_coordinates=machine_coordinates,
                pixel_coordinates=pixel_coordinates,
            )

        details_by_id[spectrum_id] = AcquisitionDetails(
            frame_id=frame_id,
            particle_id=_parse_optional_int(row.get(cnst.PAR_ID_DF_KEY)),
            spot_coordinates=spot_coordinates,
        )

    return details_by_id


def load_legacy_quantification_results_by_spectrum_id(
    data_csv_path: str,
) -> Dict[str, List[QuantificationResult]]:
    """Load quantification results keyed by spectrum id from legacy Data.csv."""
    results_by_id: Dict[str, List[QuantificationResult]] = {}
    if not data_csv_path or not os.path.exists(data_csv_path):
        return results_by_id

    try:
        data_df = pd.read_csv(data_csv_path)
    except Exception:
        return results_by_id

    el_atfr_cols = [col for col in data_df.columns if col.endswith(cnst.AT_FR_DF_KEY)]
    el_wfr_cols = [col for col in data_df.columns if col.endswith(cnst.W_FR_DF_KEY)]
    analytical_error_col = _resolve_first_present_column(data_df, [cnst.AN_ER_DF_KEY, "An Er w%"])
    r_squared_col = _resolve_first_present_column(data_df, [cnst.R_SQ_KEY, "r_sq", "R squared"])
    reduced_chi_squared_col = _resolve_first_present_column(data_df, [cnst.REDCHI_SQ_KEY, "redchi", "Reduced Chi-square"])
    if not el_atfr_cols and not el_wfr_cols and cnst.QUANT_FLAG_DF_KEY not in data_df.columns:
        return results_by_id

    for row_idx, row in data_df.iterrows():
        spectrum_id_val = row.get(cnst.SP_ID_DF_KEY, row_idx)
        if isinstance(spectrum_id_val, (int, float, np.integer)) and not pd.isna(spectrum_id_val):
            spectrum_id = str(int(spectrum_id_val))
        else:
            spectrum_id = str(spectrum_id_val)
        if not spectrum_id:
            spectrum_id = str(row_idx)

        has_quant_data = any(pd.notnull(row[col]) for col in (el_atfr_cols + el_wfr_cols))

        comp_at_fr: Dict[str, float] = {}
        comp_w_fr: Dict[str, float] = {}
        for col in el_atfr_cols:
            val = _parse_optional_float(row.get(col))
            if val is None:
                continue
            comp_at_fr[col.replace(cnst.AT_FR_DF_KEY, "")] = val / 100.0
        for col in el_wfr_cols:
            val = _parse_optional_float(row.get(col))
            if val is None:
                continue
            comp_w_fr[col.replace(cnst.W_FR_DF_KEY, "")] = val / 100.0

        analytical_error = None
        if analytical_error_col is not None:
            analytical_error = _parse_analytical_error_percent(row.get(analytical_error_col))

        r_squared = _parse_optional_float(row.get(r_squared_col)) if r_squared_col is not None else None
        reduced_chi_squared = (
            _parse_optional_float(row.get(reduced_chi_squared_col)) if reduced_chi_squared_col is not None else None
        )
        fit_result = None
        if r_squared is not None or reduced_chi_squared is not None:
            fit_result = FitResult(
                r_squared=r_squared,
                reduced_chi_squared=reduced_chi_squared,
            )

        raw_comment = row.get(cnst.COMMENTS_DF_KEY) if cnst.COMMENTS_DF_KEY in data_df.columns else None
        comment = None
        if raw_comment is not None and not pd.isna(raw_comment):
            comment = str(raw_comment).strip() or None

        quant_flag = _parse_optional_int(row.get(cnst.QUANT_FLAG_DF_KEY)) if cnst.QUANT_FLAG_DF_KEY in data_df.columns else None
        is_interrupted = not has_quant_data
        diagnostics = QuantificationDiagnostics(
            converged=False if (is_interrupted or quant_flag == -1) else True,
            interrupted=is_interrupted,
            min_background_ref_lines=_extract_min_background_ref_lines_from_comment(comment),
        )

        quant_record = QuantificationResult(
            quantification_id=0,
            quant_flag=quant_flag,
            comment=comment,
            composition_atomic_fractions=comp_at_fr or None,
            composition_weight_fractions=comp_w_fr or None,
            analytical_error=analytical_error,
            fit_result=fit_result,
            diagnostics=diagnostics,
        )
        results_by_id[spectrum_id] = [quant_record]

    return results_by_id


def build_legacy_import_quantification_config(
    *,
    sample_result_dir: str,
    ledger_configs: Any,
) -> QuantificationConfig:
    """Build legacy quantification config with standards-derived reference values."""
    quant_cfg = getattr(ledger_configs, "quant_cfg", None)
    if quant_cfg is None:
        quant_cfg = _load_legacy_quant_cfg(sample_result_dir)

    legacy_clustering_cfg = _load_legacy_clustering_cfg(sample_result_dir) or ClusteringConfig()

    method = str(getattr(quant_cfg, "method", "PB") or "PB").strip() or "PB"
    fit_tolerance = float(getattr(quant_cfg, "fit_tolerance", 1e-4) or 1e-4)
    use_instrument_background = bool(getattr(quant_cfg, "use_instrument_background", False))

    spectrum_lims_raw = getattr(quant_cfg, "spectrum_lims", [0.0, 0.0])
    if not isinstance(spectrum_lims_raw, (list, tuple)) or len(spectrum_lims_raw) != 2:
        spectrum_lims_raw = [0.0, 0.0]
    spectrum_lims = [float(spectrum_lims_raw[0]), float(spectrum_lims_raw[1])]

    options = {
        "method": method,
        "fit_tolerance": fit_tolerance,
        "use_instrument_background": use_instrument_background,
        "spectrum_lims": spectrum_lims,
    }

    reference_values_by_el_line: Dict[str, Any] = {}
    if method == "PB":
        try:
            import autoemx.calibrations as calibs

            microscope_id = str(getattr(ledger_configs.microscope_cfg, "ID", "")).strip()
            if microscope_id:
                calibs.load_microscope_calibrations(
                    microscope_ID=microscope_id,
                    meas_mode=ledger_configs.measurement_cfg.mode,
                    load_detector_channel_params=False,
                )

            use_project_specific_std_dict = bool(
                getattr(quant_cfg, "use_project_specific_std_dict", False)
            )
            std_f_dir = str(Path(sample_result_dir).parent) if use_project_specific_std_dict else None
            standards_by_mode, _ = calibs.load_standards(
                ledger_configs.measurement_cfg.type,
                ledger_configs.measurement_cfg.beam_energy_keV,
                std_f_dir=std_f_dir,
            )
            standards_by_line = standards_by_mode.get(ledger_configs.measurement_cfg.mode, {})
            relevant_elements = set(ledger_configs.sample_cfg.elements) | set(ledger_configs.sample_substrate_cfg.elements)
            reference_values_by_el_line = _extract_reference_values_for_quantification(
                standards_by_line,
                relevant_elements,
            )
        except Exception:
            reference_values_by_el_line = {}

    return QuantificationConfig(
        quantification_id=0,
        label="Legacy import",
        sample_elements=list(ledger_configs.sample_cfg.elements),
        substrate_elements=list(ledger_configs.sample_substrate_cfg.elements),
        options=options,
        reference_values_by_el_line=dict(sorted(reference_values_by_el_line.items())),
        reference_lines_by_element=QuantificationConfig.derive_reference_lines_by_element(
            reference_values_by_el_line=reference_values_by_el_line,
            preferred_lines=["Ka1", "La1", "Ma1", "Mz1"],
        ),
        clustering_configs=[
            ClusteringConfig(
                clustering_id=0,
                method=str(getattr(legacy_clustering_cfg, "method", "kmeans")),
                features=str(getattr(legacy_clustering_cfg, "features", "")),
                k_forced=getattr(
                    legacy_clustering_cfg,
                    "k_forced",
                    getattr(legacy_clustering_cfg, "k", None),
                ),
                k_finding_method=str(getattr(legacy_clustering_cfg, "k_finding_method", "silhouette")),
                max_k=int(getattr(legacy_clustering_cfg, "max_k", 6)),
                ref_formulae=list(getattr(legacy_clustering_cfg, "ref_formulae", []) or []),
                do_matrix_decomposition=bool(getattr(legacy_clustering_cfg, "do_matrix_decomposition", True)),
                max_analytical_error_percent=getattr(legacy_clustering_cfg, "max_analytical_error_percent", 5.0),
                min_bckgrnd_cnts=getattr(legacy_clustering_cfg, "min_bckgrnd_cnts", 5),
                quant_flags_accepted=list(getattr(legacy_clustering_cfg, "quant_flags_accepted", [0, -1]) or []),
            )
        ],
        active_clustering_cfg_index=0,
    )
