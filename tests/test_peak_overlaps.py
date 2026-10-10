#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the peak overlap warnings."""
import json

from autoemx.core.fitter.peaks import OVERLAP_SEPARATION_SIGMAS
from autoemx.core.quantifier.peak_overlaps import (
    PEAK_OVERLAPS_FILE,
    format_peak_overlaps_warning,
    get_peak_overlaps,
)

ENERGY_RANGE = (0.14, 11)


def _summary(overlaps):
    """{quantification line: [overlapping lines]} of the overlaps."""
    return {
        f"{o['line']['element']} {o['line']['line']}": [f"{x['element']} {x['line']}" for x in o["overlapping"]]
        for o in overlaps
    }


def test_table_defines_severities_per_method():
    with open(PEAK_OVERLAPS_FILE, encoding="utf-8") as file:
        eds = json.load(file)["methods"]["EDS"]
    # Severity is defined in detector sigmas, the 'significant' threshold being the fitter's overlap criterion
    assert eds["severity_thresholds_sigma"] == {"severe": 2, "significant": OVERLAP_SEPARATION_SIGMAS}


def test_mos2_severe_overlap():
    overlaps = get_peak_overlaps("EDS", ["Mo", "S"], beam_energy_keV=15, energy_range_keV=ENERGY_RANGE)
    assert _summary(overlaps) == {"Mo La1": ["S Ka1"], "S Ka1": ["Mo La1"]}
    assert all(o["severity"] == "severe" for o in overlaps)
    assert overlaps[0]["overlapping"][0]["separation_sigma"] < 1
    warning = format_peak_overlaps_warning(overlaps)
    assert "S may be inaccurately quantified: its reference peak S Ka1 (2.307 keV) overlaps with Mo La1" in warning
    assert "whose area is also fitted freely" in warning


def test_no_warning_for_methods_without_criteria():
    assert get_peak_overlaps("WDS", ["Mo", "S"], beam_energy_keV=15) == []


def test_only_quantification_lines_of_sample_elements_are_warned_about():
    # At 15 keV Ta is quantified with Ta La: Sr La overlaps Ta Ma, whose area is free, but not the reverse
    overlaps = get_peak_overlaps("EDS", ["Sr", "Ta", "O"], ["C"], 15, ENERGY_RANGE)
    assert _summary(overlaps) == {"Sr La1": ["Ta Ma1"]}
    assert overlaps[0]["severity"] == "significant"
    # At 20 keV Cu is quantified with Cu Ka and Pr with Pr La, so Cu La / Pr Ma does not matter
    assert get_peak_overlaps("EDS", ["Cu", "Pr", "O"], [], 20, ENERGY_RANGE) == []
    # At 5 keV the K and L lines are not excited, so the low-energy lines are used for quantification
    overlaps = get_peak_overlaps("EDS", ["Cu", "Pr", "O"], [], 5, ENERGY_RANGE)
    assert {"Cu La1", "Pr Ma1"} <= set(_summary(overlaps))


def test_free_area_lines_of_substrate_and_non_quantification_lines():
    # Mo La of the sample overlaps S Ka of the substrate, whose area is free; substrate lines are not warned about
    assert _summary(get_peak_overlaps("EDS", ["Mo", "O"], ["S"], 15, ENERGY_RANGE)) == {"Mo La1": ["S Ka1"]}
    # Cl Ll is the free L reference line of Cl, overlapping B Ka
    overlaps = get_peak_overlaps("EDS", ["B", "N"], ["Cl"], 15, ENERGY_RANGE)
    assert _summary(overlaps) == {"B Ka1": ["Cl Ll"]}
    # Lines with a tied area (e.g. Pb Mz, tied to Pb Ma) are not considered
    overlaps = get_peak_overlaps("EDS", ["Sr", "O"], ["Pb"], 15, ENERGY_RANGE)
    assert all("Pb Mz1" not in lines for lines in _summary(overlaps).values())


def test_free_area_el_lines():
    # Ge Lb1 is fitted with a free area by default, and overlaps Mg Ka
    assert "Ge Lb1" in _summary(get_peak_overlaps("EDS", ["Mg", "O"], ["Ge"], 15, ENERGY_RANGE))["Mg Ka1"]
    overlaps = get_peak_overlaps("EDS", ["Mg", "O"], ["Ge"], 15, ENERGY_RANGE, free_area_el_lines=[])
    assert "Ge Lb1" not in _summary(overlaps)["Mg Ka1"]
    # Lines made free by the user are considered, also within the same element (Fe is quantified with Fe La at 5 keV)
    assert get_peak_overlaps("EDS", ["Fe"], [], 5, ENERGY_RANGE, free_area_el_lines=[]) == []
    overlaps = get_peak_overlaps("EDS", ["Fe"], [], 5, ENERGY_RANGE, free_area_el_lines=["Fe_Lb1"])
    assert _summary(overlaps) == {"Fe La1": ["Fe Lb1"]}


def test_severity_depends_on_energy():
    # ~170 eV is still a significant overlap at ~10 keV, where the detector peaks are broad
    overlaps = get_peak_overlaps("EDS", ["Ge", "Au"], [], 20, (0.14, 15))
    ge = next(o for o in overlaps if o["line"]["line"] == "Ka1")
    assert ge["overlapping"][0]["element"] == "Au" and ge["severity"] == "significant"
    # ~80 eV is resolved at ~1 keV, where the detector peaks are narrow
    overlaps = get_peak_overlaps("EDS", ["Cu", "Zn"], [], 5, ENERGY_RANGE)
    assert "Zn La1" not in _summary(overlaps).get("Cu La1", [])


def test_web_overlaps_and_export():
    import numpy as np

    from autoemx.web.exports import format_composition_txt
    from autoemx.web.pipeline import SpectrumFitResult, find_peak_overlaps

    overlaps = find_peak_overlaps(["Mo", "Pb", "O"], ["C", "S"])
    assert "S Ka1" in _summary(overlaps)["Mo La1"]
    assert find_peak_overlaps(["Bi", "Fe", "O"], ["C", "O", "Al"]) == []

    result = SpectrumFitResult(
        filename="x.msa", energy_keV=np.array([]), counts=np.array([]), fit=np.array([]),
        background=np.array([]), composition_at={}, composition_wt={}, peak_overlaps=overlaps,
    )
    txt = format_composition_txt(result)
    assert "# WARNING: Peak overlaps may cause inaccurate quantification of Mo, Pb:" in txt
    assert "Mo may be inaccurately quantified: its reference peak Mo La1 (2.293 keV) overlaps with" in txt
