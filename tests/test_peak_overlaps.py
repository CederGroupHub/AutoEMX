#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the peak overlap table and warnings."""
import json

from autoemx.core.fitter.peaks import OVERLAP_SEPARATION_SIGMAS
from autoemx.core.quantifier.peak_overlaps import (
    PEAK_OVERLAPS_FILE,
    format_peak_overlaps_warning,
    get_peak_overlaps,
)

ENERGY_RANGE = (0.14, 11)


def _pair_names(overlaps):
    return [{f"{line['element']} {line['line']}" for line in o["lines"]} for o in overlaps]


def test_table_defines_severities_per_method():
    with open(PEAK_OVERLAPS_FILE, encoding="utf-8") as file:
        eds = json.load(file)["methods"]["EDS"]
    # Severity is defined in detector sigmas, the 'significant' threshold being the fitter's overlap criterion
    assert eds["severity_thresholds_sigma"] == {"severe": 2, "significant": OVERLAP_SEPARATION_SIGMAS}
    assert all(p["separation_eV"] <= eds["max_candidate_separation_eV"] for p in eds["pairs"])


def test_mos2_severe_overlap():
    overlaps = get_peak_overlaps("EDS", ["Mo", "S"], beam_energy_keV=15, energy_range_keV=ENERGY_RANGE)
    assert _pair_names(overlaps) == [{"Mo La1", "S Ka1"}]
    assert overlaps[0]["severity"] == "severe"
    assert overlaps[0]["separation_sigma"] < 1
    assert overlaps[0]["affected_elements"] == ["Mo", "S"]
    assert "Severe overlap" in format_peak_overlaps_warning(overlaps)


def test_no_warning_for_methods_without_table():
    assert get_peak_overlaps("WDS", ["Mo", "S"], beam_energy_keV=15) == []


def test_overlap_ignored_when_both_elements_quantified_with_other_lines():
    # At 20 keV Cu is quantified with Cu Ka and Pr with Pr La, so Cu La / Pr Ma does not matter
    assert get_peak_overlaps("EDS", ["Cu", "Pr", "O"], ["C"], 20, ENERGY_RANGE) == []
    # At 5 keV the K and L lines are not excited, so the low-energy lines are used for quantification
    overlaps = get_peak_overlaps("EDS", ["Cu", "Pr", "O"], ["C"], 5, ENERGY_RANGE)
    assert {"Cu La1", "Pr Ma1"} in _pair_names(overlaps)


def test_only_affected_elements_are_reported():
    # Ta is quantified with Ta La at 15 keV, while Sr La is the only line available for Sr
    overlaps = get_peak_overlaps("EDS", ["Sr", "Ta", "O"], ["C"], 15, ENERGY_RANGE)
    assert _pair_names(overlaps) == [{"Ta Ma1", "Sr La1"}]
    assert overlaps[0]["severity"] == "significant"
    assert overlaps[0]["affected_elements"] == ["Sr"]


def test_severity_depends_on_energy():
    # ~170 eV is still a significant overlap at ~10 keV, where the detector peaks are broad
    overlaps = get_peak_overlaps("EDS", ["Ge", "Au"], beam_energy_keV=20, energy_range_keV=(0.14, 15))
    assert {"Ge Ka1", "Au La1"} in _pair_names(overlaps)
    assert overlaps[_pair_names(overlaps).index({"Ge Ka1", "Au La1"})]["severity"] == "significant"
    # ~80 eV is resolved at ~1 keV, where the detector peaks are narrow
    overlaps = get_peak_overlaps("EDS", ["Cu", "Zn"], beam_energy_keV=5, energy_range_keV=ENERGY_RANGE)
    assert {"Cu La1", "Zn La1"} not in _pair_names(overlaps)


def test_substrate_elements_are_considered():
    # Mo La of the sample overlaps S Ka of the substrate
    overlaps = get_peak_overlaps("EDS", ["Mo", "O"], ["S"], 15, ENERGY_RANGE)
    assert overlaps[0]["affected_elements"] == ["Mo"]
    # Overlaps between substrate elements only are not reported
    assert get_peak_overlaps("EDS", ["Fe", "O"], ["Mo", "S"], 15, ENERGY_RANGE) == []


def test_web_overlaps_and_export():
    import numpy as np

    from autoemx.web.exports import format_composition_txt
    from autoemx.web.pipeline import SpectrumFitResult, find_peak_overlaps

    overlaps = find_peak_overlaps(["Mo", "Pb", "O"], ["C", "S"])
    assert {"Mo La1", "S Ka1"} in _pair_names(overlaps)
    assert find_peak_overlaps(["Bi", "Fe", "O"], ["C", "O", "Al"]) == []

    result = SpectrumFitResult(
        filename="x.msa", energy_keV=np.array([]), counts=np.array([]), fit=np.array([]),
        background=np.array([]), composition_at={}, composition_wt={}, peak_overlaps=overlaps,
    )
    txt = format_composition_txt(result)
    assert "# WARNING: Peak overlaps may cause inaccurate quantification of Mo, Pb:" in txt
    assert "Severe overlap: Mo La1 (2.293 keV) and S Ka1 (2.307 keV)" in txt
