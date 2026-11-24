#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line used to quantify an element when its preferred reference line has no standard.

E.g. Br at 20 keV: Br_Ka1 (11.9 keV) is the preferred line, but with standards of Br_La1 only, Br is quantified with
Br_La1 (with a warning), and the quantification run records Br_La1. Only an element with no reference line with a
standard cannot be quantified. No spectra are quantified.
"""

from __future__ import annotations

from types import SimpleNamespace

from autoemx.core.quantifier.quant_flags import check_fit_quant_validity
from autoemx.core.quantifier.quantifier import XSp_Quantifier as Q

BR_LINES = {"Br_Ka1": 11.92, "Br_La1": 1.48}


def test_preferred_line_with_standard():
    line, note = Q.select_quant_line(BR_LINES, 20, has_standard=lambda l: True)
    assert (line, note) == ("Br_Ka1", None)


def test_fallback_to_line_with_standard():
    line, note = Q.select_quant_line(BR_LINES, 20, has_standard=lambda l: l == "Br_La1",
                                     areas={"Br_Ka1": 50.0, "Br_La1": 10.0})
    assert line == "Br_La1"
    assert "Br_La1 used instead of Br_Ka1" in note


def test_no_line_with_standard_keeps_preferred():
    # Quantification then fails, naming the lines (see _get_k_ratios)
    line, note = Q.select_quant_line(BR_LINES, 20, has_standard=lambda l: False)
    assert (line, note) == ("Br_Ka1", None)


def test_without_standards_unchanged():
    # Largest peak among the ideal candidates (only Br_Ka1 is above 2 keV with overvoltage > 1.65)
    line, note = Q.select_quant_line(BR_LINES, 20, areas={"Br_Ka1": 5.0, "Br_La1": 100.0})
    assert (line, note) == ("Br_Ka1", None)


def test_reference_lines_recorded_for_run():
    lines = Q.reference_lines_for_elements(["Br", "Zn", "O"], {"Br_La1", "Zn_Ka1", "Zn_La1", "O_Ka1"},
                                           beam_energy=20, energy_range_keV=(0.1, 14.5))
    assert lines == {"Br": "Br_La1", "O": "O_Ka1", "Zn": "Zn_Ka1"}
    # Element without any line with a standard: not recorded
    assert "Br" not in Q.reference_lines_for_elements(["Br"], {"Zn_Ka1"}, 20, (0.1, 14.5))
    notes = {}
    Q.reference_lines_for_elements(["Br", "Zn"], {"Br_La1", "Zn_Ka1"}, 20, (0.1, 14.5), fallback_notes=notes)
    assert list(notes) == ["Br"] and "Br_La1 used instead of Br_Ka1" in notes["Br"]


def test_interrupted_fit_comment_gives_reason():
    kwargs = dict(is_quant_fit_valid=False, bad_quant_flag=None, min_bckgrnd_ref_lines=None,
                  detectable_els_substrate=[], min_bckgrnd_cnts=None)
    flag, comment = check_fit_quant_validity(quantifier=SimpleNamespace(fit_error="EDSError: no standard"), **kwargs)
    assert (flag, comment) == (9, "Fit interrupted: EDSError: no standard")
    assert check_fit_quant_validity(quantifier=None, **kwargs)[1] == "Fit interrupted for unknown reasons"
