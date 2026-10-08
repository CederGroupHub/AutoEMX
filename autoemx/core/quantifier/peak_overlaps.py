#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peak overlap warnings
=====================

Identifies overlaps between the freely fitted X-ray lines of different elements that may
compromise the quantification of a sample, based on the table in
autoemx/data/Xray_lines/common_peak_overlaps.json (see make_common_peak_overlaps_file.py).

The overlap table and its severity categories are defined per measurement method. Methods
without an entry in the table (anything other than 'EDS', currently) produce no warnings.

For EDS, the severity of an overlap depends on the separation of the two lines in units of
sigma_eff = sqrt((sigma1^2 + sigma2^2) / 2), with sigma the detector resolution at each line energy,
computed with the calibration of the microscope used (DetectorResponseFunction.det_sigma).
"""
import importlib
import json
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import autoemx.config.defaults as dflt
from autoemx._logging import get_logger
from autoemx.core.fitter.detector_response import DetectorResponseFunction
from autoemx.core.quantifier.quantifier import XSp_Quantifier
from autoemx.data.Xray_lines import SCRIPT_DIR as XRAY_LINES_DIR, get_el_xray_lines
from autoemx.utils.helper import print_double_separator

logger = get_logger(__name__)

PEAK_OVERLAPS_FILE = os.path.join(XRAY_LINES_DIR, "common_peak_overlaps.json")

# Lines are fitted only if the overvoltage exceeds this value (see XSp_Fitter._define_xray_lines)
MIN_FIT_OVERVOLTAGE = 1.2


@lru_cache(maxsize=None)
def _load_method_table(meas_type: str) -> Optional[dict]:
    """Return the overlap table of a measurement method, or None if the method has none."""
    with open(PEAK_OVERLAPS_FILE, "r", encoding="utf-8") as file:
        methods = json.load(file)["methods"]
    return methods.get(meas_type)


@lru_cache(maxsize=None)
def _load_detector_params(microscope_ID: str) -> Tuple[float, float, float]:
    """Detector resolution parameters (conv_eff, elec_noise, F) from the calibration module of a microscope."""
    mod = importlib.import_module(f"autoemx.calibrations.{microscope_ID}.XS_calibrations")
    return mod.conv_eff, mod.elec_noise, mod.F


def _separation_sigma(energy_a: float, energy_b: float, detector_params: Tuple[float, float, float]) -> float:
    """Separation of two lines (keV) in units of the RMS detector sigma at their energies."""
    sigma_a, sigma_b = (DetectorResponseFunction.det_sigma(en, *detector_params) for en in (energy_a, energy_b))
    return abs(energy_a - energy_b) / np.sqrt((sigma_a ** 2 + sigma_b ** 2) / 2)


def _severity(separation_sigma: float, thresholds: Dict[str, float]) -> Optional[str]:
    """Severity category of an overlap, from the thresholds in increasing order (None if not overlapping)."""
    for label, threshold in thresholds.items():
        if separation_sigma < threshold:
            return label
    return None


def _is_line_fitted(
    energy: float,
    beam_energy_keV: Optional[float],
    energy_range_keV: Optional[Tuple[float, float]],
) -> bool:
    """Whether a line of given energy (keV) is included in the fit."""
    if beam_energy_keV is not None and beam_energy_keV / energy <= MIN_FIT_OVERVOLTAGE:
        return False
    if energy_range_keV is not None and not energy_range_keV[0] <= energy <= energy_range_keV[1]:
        return False
    return True


def _get_candidate_quant_lines(
    el_lines: Dict[str, float],
    beam_energy_keV: Optional[float],
) -> set:
    """
    Lines that may be used to quantify an element, following XSp_Quantifier: the fitted reference
    lines above the ideal energy and overvoltage thresholds if any, otherwise all fitted reference lines.
    The choice among multiple candidates depends on fitted peak areas, so all candidates are returned.
    """
    ref_lines = {line: en for line, en in el_lines.items() if line in XSp_Quantifier.xray_quant_ref_lines}
    if beam_energy_keV is None:
        return set(ref_lines)
    ideal_lines = {
        line for line, en in ref_lines.items()
        if en > XSp_Quantifier.ideal_ref_line_energy_threshold
        and beam_energy_keV / en > XSp_Quantifier.ideal_ref_line_overvoltage
    }
    return ideal_lines or set(ref_lines)


def get_peak_overlaps(
    meas_type: str,
    els_sample: Iterable[str],
    els_substrate: Iterable[str] = (),
    beam_energy_keV: Optional[float] = None,
    energy_range_keV: Optional[Tuple[float, float]] = None,
    microscope_ID: Optional[str] = None,
) -> List[dict]:
    """
    Find the overlaps between fitted lines that may compromise the quantification of sample elements.

    An overlap is reported when both lines are fitted (within the energy range and excited by the beam),
    both elements are among the sample or substrate elements, and at least one of the two lines may be
    used to quantify a sample element.

    Parameters
    ----------
    meas_type : str
        Measurement method (e.g. 'EDS'). Methods without an overlap table return no overlaps.
    els_sample, els_substrate : iterable of str
        Element symbols of the sample and of the substrate.
    beam_energy_keV : float, optional
        Beam energy. If None, lines are not filtered by overvoltage and all reference lines of an
        element are considered as possible quantification lines.
    energy_range_keV : (float, float), optional
        Energy range of the fitted spectrum. If None, lines are not filtered by energy.
    microscope_ID : str, optional
        Microscope whose detector calibration defines the resolution. Defaults to the default microscope.

    Returns
    -------
    list of dict
        One entry per overlapping pair, sorted by increasing separation in sigmas, with keys 'lines'
        (list of dicts with 'element', 'line', 'energy_keV'), 'separation_eV', 'separation_sigma',
        'severity', 'affected_elements' (sample elements whose quantification may be inaccurate) and 'sources'.
    """
    table = _load_method_table(meas_type)
    if table is None:
        return []

    detector_params = _load_detector_params(microscope_ID or dflt.microscope_ID)
    thresholds = table["severity_thresholds_sigma"]

    els_sample = {el for el in els_sample if el}
    els_all = els_sample | {el for el in els_substrate if el}

    # Overlapping pairs of fitted lines of the relevant elements
    pairs = [
        pair for pair in table["pairs"]
        if all(
            line["element"] in els_all and _is_line_fitted(line["energy_keV"], beam_energy_keV, energy_range_keV)
            for line in pair["lines"]
        )
    ]

    # Lines that may be used to quantify each sample element
    candidate_quant_lines = {}
    for el in els_sample:
        try:
            all_lines = get_el_xray_lines(el)
        except (ValueError, KeyError):
            continue
        fitted_lines = {
            line: float(info["energy (keV)"]) for line, info in all_lines.items()
            if _is_line_fitted(float(info["energy (keV)"]), beam_energy_keV, energy_range_keV)
        }
        candidate_quant_lines[el] = _get_candidate_quant_lines(fitted_lines, beam_energy_keV)

    overlaps = []
    for pair in pairs:
        line_a, line_b = pair["lines"]
        separation_sigma = _separation_sigma(line_a["energy_keV"], line_b["energy_keV"], detector_params)
        severity = _severity(separation_sigma, thresholds)
        if severity is None:
            continue
        affected = sorted({
            line["element"] for line in pair["lines"]
            if line["line"] in candidate_quant_lines.get(line["element"], ())
        })
        if not affected:
            continue
        overlaps.append({
            "lines": [{k: line[k] for k in ("element", "line", "energy_keV")} for line in pair["lines"]],
            "separation_eV": pair["separation_eV"],
            "separation_sigma": round(float(separation_sigma), 2),
            "severity": severity,
            "affected_elements": affected,
            "sources": pair["sources"],
        })
    overlaps.sort(key=lambda o: o["separation_sigma"])
    return overlaps


def format_line(line: dict) -> str:
    """Format a line entry, e.g. 'S Ka1 (2.307 keV)'."""
    return f"{line['element']} {line['line']} ({line['energy_keV']:.3f} keV)"


def format_overlap(overlap: dict) -> str:
    """Format an overlap, e.g. 'Severe overlap: Mo La1 (2.293 keV) and S Ka1 (2.307 keV), ΔE = 14 eV (0.4σ)'."""
    line_a, line_b = overlap["lines"]
    return (
        f"{overlap['severity'].capitalize()} overlap: {format_line(line_a)} and {format_line(line_b)}, "
        f"ΔE = {overlap['separation_eV']:.0f} eV ({overlap['separation_sigma']:.1f}σ)"
    )


def affected_elements(overlaps: Sequence[dict]) -> List[str]:
    """Sample elements whose quantification may be compromised by the overlaps."""
    return sorted({el for o in overlaps for el in o["affected_elements"]})


def format_peak_overlaps_warning(overlaps: Sequence[dict]) -> str:
    """Format the warning message for a list of overlaps returned by get_peak_overlaps."""
    msg_lines = [f"⚠️ Peak overlaps may cause inaccurate quantification of {', '.join(affected_elements(overlaps))}:"]
    msg_lines.extend(f"   • {format_overlap(o)}" for o in overlaps)
    return "\n".join(msg_lines)


def warn_peak_overlaps(
    meas_type: str,
    els_sample: Iterable[str],
    els_substrate: Iterable[str] = (),
    beam_energy_keV: Optional[float] = None,
    energy_range_keV: Optional[Tuple[float, float]] = None,
    microscope_ID: Optional[str] = None,
) -> List[dict]:
    """Log a warning, enclosed in double separators, listing the overlaps found by get_peak_overlaps."""
    overlaps = get_peak_overlaps(
        meas_type, els_sample, els_substrate, beam_energy_keV, energy_range_keV, microscope_ID
    )
    if overlaps:
        print_double_separator()
        logger.warning(format_peak_overlaps_warning(overlaps))
        print_double_separator()
    return overlaps
