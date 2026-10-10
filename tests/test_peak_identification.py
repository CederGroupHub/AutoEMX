#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the automatic element identification."""
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls

from ci_paths import INPUTS_DIR, WULFENITE_MINI_ID
from autoemx.config.schema_models.ledger import SampleLedger
from autoemx.core.fitter import peak_identification as pid

MICROSCOPE_ID = "PhenomXL"
BEAM_E = 15.0
ENERGY = -0.034245 + 0.009972 * np.arange(14, 1100)
SIGMA = pid.get_sigma_function(MICROSCOPE_ID)
RNG_SEED = 0


def _family_spectrum(elements_counts):
    """
    Noise-free peaks of the line families of elements above 1 keV, {element: counts of each family reference
    line} (low-energy families are strongly absorbed in real spectra).
    """
    out = np.zeros_like(ENERGY)
    for el, counts in elements_counts.items():
        for ref, lines in pid.element_line_families(el, BEAM_E, (ENERGY[0], ENERGY[-1])).items():
            if dict((ln, en) for ln, en, _ in lines)[ref] < 1:
                continue
            out += pid.gaussian_template(ENERGY, [(en, w * counts) for _, en, w in lines], SIGMA)
    return out


def _background():
    return 3000 * np.exp(-ENERGY / 2.5) + 30


def _template_fit_function(y):
    """Fake fit function: weighted NNLS of a smooth background and the line families of the elements."""
    bg_basis = [np.exp(-ENERGY / 2.5), np.ones_like(ENERGY), ENERGY]

    def fit(elements):
        cols, names = list(bg_basis), [None] * len(bg_basis)
        for el in elements:
            for ref, lines in pid.element_line_families(el, BEAM_E, (ENERGY[0], ENERGY[-1])).items():
                cols.append(pid.gaussian_template(ENERGY, [(en, w) for _, en, w in lines], SIGMA))
                names.append(f"{el}_{ref}")
        A = np.column_stack(cols)
        w = 1 / np.sqrt(np.maximum(y, 1))
        # Background terms may be negative: split them into positive and negative parts
        A_full = np.column_stack([A, -A[:, :len(bg_basis)]])
        coef, _ = nnls(A_full * w[:, None], y * w)
        model = A_full @ coef
        n_bg = len(bg_basis)
        areas = {name: float(c) for name, c in zip(names, coef) if name}
        components = {name: coef[i] * cols[i] for i, name in enumerate(names) if name}
        background = A_full[:, :n_bg] @ coef[:n_bg] - A[:, :n_bg] @ coef[len(cols):]
        return pid.FitOutcome(tuple(sorted(elements)), model, pid.poisson_deviance(y, model), areas,
                              components, background)

    return fit


def _identify(y, fixed=(), **kwargs):
    return pid.identify_elements(y, ENERGY, _template_fit_function(y), BEAM_E, SIGMA, fixed_elements=fixed, **kwargs)


# =============================================================================
# Filter and residual peaks
# =============================================================================
def test_top_hat_removes_linear_background():
    F = pid.top_hat_matrix(ENERGY, SIGMA)
    active = np.any(F != 0, axis=1)
    assert active.sum() > 0.9 * len(ENERGY)
    np.testing.assert_allclose(F.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(F @ (5 + 3 * ENERGY), 0, atol=1e-9)


def test_residual_peak_energy_and_area():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + pid.gaussian_template(ENERGY, [(6.4, 2000)], SIGMA)).astype(float)
    res = pid._ResidualAnalysis(y, None, ENERGY, pid.top_hat_matrix(ENERGY, SIGMA), SIGMA)
    peaks = res.find_peaks(pid.PeakIDSettings())
    assert len(peaks) == 1
    assert abs(peaks[0].energy_keV - 6.4) < 0.02
    assert peaks[0].net_area == pytest.approx(2000, rel=0.15)


def test_overlap_partners_from_table():
    partners = {el for el, _, _ in pid._overlap_partners("S", "Ka1", SIGMA)}
    assert {"Mo", "Pb"} <= partners


def test_confirmation_lines_required():
    # Fe L lines alone (no Fe K at 6.4 keV, well excited at 15 keV) must not be identified as Fe
    rng = np.random.default_rng(RNG_SEED)
    fe_l = {ref: lines for ref, lines in pid.element_line_families("Fe", BEAM_E, (ENERGY[0], ENERGY[-1])).items() if ref == "La1"}
    peaks = pid.gaussian_template(ENERGY, [(en, w * 5000) for _, en, w in fe_l["La1"]], SIGMA)
    y = rng.poisson(_background() + peaks).astype(float)
    res = pid._ResidualAnalysis(y, _background(), ENERGY, pid.top_hat_matrix(ENERGY, SIGMA), SIGMA)
    s = pid._screen_candidate("Fe", "La1", "La1", 0.705, res, BEAM_E, (ENERGY[0], ENERGY[-1]), pid.PeakIDSettings())
    assert not s.passed
    assert "confirmation" in s.reason


# =============================================================================
# Identification loop (template fit function)
# =============================================================================
def test_identifies_elements_from_scratch():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Cu": 8000})).astype(float)
    result = _identify(y)
    assert set(result.new_elements) == {"Fe", "Cu"}
    assert set(result.strong_new_elements) == {"Fe", "Cu"}
    assert result.possible_new_elements == []
    assert result.unexplained_peaks == []


def test_peaks_assigned_before_starting_fit():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Cu": 8000})).astype(float)
    F = pid.top_hat_matrix(ENERGY, SIGMA)
    table = pid._strong_lines_table(BEAM_E, ENERGY[0], ENERGY[-1], pid.PeakIDSettings().min_line_weight)
    assigned = pid._assign_peaks(y, ENERGY, F, SIGMA, BEAM_E, (ENERGY[0], ENERGY[-1]), table,
                                 pid.PeakIDSettings(), known_elements=[], excluded=set())
    assert [sc.element for sc in assigned] == ["Fe", "Cu"]
    # With no known sample element, the starting fit includes the assigned elements
    result = _identify(y)
    assert all(e.from_peak_assignment for e in result.elements if e.element in ("Fe", "Cu"))
    # Peaks of known (substrate) elements are not assigned
    assert pid._assign_peaks(y, ENERGY, F, SIGMA, BEAM_E, (ENERGY[0], ENERGY[-1]), table, pid.PeakIDSettings(),
                             known_elements=["Fe"], excluded=set())[0].element == "Cu"


def test_fixed_elements_are_not_reidentified():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Cu": 8000})).astype(float)
    result = _identify(y, fixed=["Fe"])
    assert result.new_elements == ["Cu"]
    assert [e.element for e in result.elements if e.status == "fixed"] == ["Fe"]


def test_nothing_to_identify():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000})).astype(float)
    assert _identify(y, fixed=["Fe"]).new_elements == []


def test_sulfur_not_confused_with_mo_or_pb():
    # S Ka overlaps Mo La and Pb Ma; the other lines of Mo and Pb (Mo Lb1, Pb Mb, Pb La) are absent
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"S": 20000, "Fe": 10000})).astype(float)
    result = _identify(y, fixed=["Fe"])
    assert result.new_elements == ["S"]


def test_excluded_and_standards_reporting():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Cu": 8000})).astype(float)
    result = _identify(y, fixed=["Fe"], has_standard=lambda el: el != "Cu")
    assert result.unquantifiable_new_elements == ["Cu"]
    assert "cannot be quantified" in result.summary()
    assert _identify(y, fixed=["Fe"], excluded_elements=["Cu"]).new_elements != ["Cu"]


def test_priority_pool_saves_candidate_tests():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Cu": 8000, "Zn": 6000})).astype(float)
    unbiased = _identify(y, fixed=["Fe"])
    pooled = _identify(y, fixed=["Fe"], priority_elements=["Cu", "Zn"])
    assert set(pooled.new_elements) == set(unbiased.new_elements) == {"Cu", "Zn"}
    assert all(e.from_pool for e in pooled.elements if e.status == "identified")
    assert pooled.n_candidate_tests < unbiased.n_candidate_tests


def test_wrong_pool_element_falls_back_to_full_search():
    # Mo in the pool cannot explain a S peak (Mo Lb1 is missing), so the full search finds S
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"S": 20000, "Fe": 10000})).astype(float)
    result = _identify(y, fixed=["Fe"], priority_elements=["Mo"])
    assert result.new_elements == ["S"]
    assert not next(e for e in result.elements if e.element == "S").from_pool


# =============================================================================
# Identification with XSp_Fitter on real spectra (2 summed spectra of wulfenite, PbMoO4)
# =============================================================================
def _wulfenite_sum():
    sample_dir = Path(INPUTS_DIR) / WULFENITE_MINI_ID
    ledger = SampleLedger.from_json_file(str(sample_dir / "ledger.json"))
    spectrum = sum(
        np.asarray(ledger._load_counts_from_pointer_file(sample_dir / e.spectrum_relpath), dtype=float)
        for e in ledger.spectra
    )
    mc = ledger.configs.microscope_cfg
    energy = mc.energy_zero + mc.bin_width * np.arange(len(spectrum))
    return spectrum[14:1100], energy[14:1100], ledger, float(spectrum.sum())


def _identify_wulfenite(fixed, inject=None):
    y, energy, ledger, tot_counts = _wulfenite_sum()
    mc, me = ledger.configs.microscope_cfg, ledger.configs.measurement_cfg
    if inject:
        rng = np.random.default_rng(RNG_SEED)
        for el, counts in inject.items():
            for lines in pid.element_line_families(el, me.beam_energy_keV, (energy[0], energy[-1])).values():
                y = y + rng.poisson(pid.gaussian_template(energy, [(en, w * counts) for _, en, w in lines], SIGMA))
    substrate = ["C", "O", "Al"]
    fit = pid.make_xsp_fit_function(
        y, energy, (14, 1100), mc.ID, me.mode, mc.energy_zero, mc.bin_width,
        me.beam_energy_keV, me.emergence_angle, tot_counts, substrate_elements=substrate, is_particle=True,
    )
    return pid.identify_elements(y, energy, fit, me.beam_energy_keV, pid.get_sigma_function(mc.ID),
                                 fixed_elements=fixed, substrate_elements=substrate)


def test_wulfenite_identification_from_scratch():
    result = _identify_wulfenite(fixed=[])
    # O is a substrate element, so it cannot be attributed to the sample
    assert set(result.new_elements) == {"Pb", "Mo"}
    # The overlap is reported on the element identified second
    notes = {e.element: e.overlaps_with for e in result.elements}
    assert any(o.startswith("Pb") for o in notes["Mo"]) or any(o.startswith("Mo") for o in notes["Pb"])


def test_wulfenite_missing_element_found():
    assert _identify_wulfenite(fixed=["Pb", "Mo", "O"]).new_elements == []
    assert _identify_wulfenite(fixed=["Pb", "Mo", "O"], inject={"Fe": 3000}).new_elements == ["Fe"]


# =============================================================================
# Residual requirement, detector artifacts, background bands
# =============================================================================
def test_residual_peak_check():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 5000})).astype(float)
    without = pid.FitOutcome((), _background(), 0.0, {}, {}, _background())
    F = pid.top_hat_matrix(ENERGY, SIGMA)
    sig_fe, ok_fe = pid._residual_peak_check("Fe", without, y, ENERGY, F, SIGMA, BEAM_E, pid.PeakIDSettings())
    sig_cu, ok_cu = pid._residual_peak_check("Cu", without, y, ENERGY, F, SIGMA, BEAM_E, pid.PeakIDSettings())
    assert ok_fe and sig_fe > 20
    assert not ok_cu


def test_background_band_ratios_detect_wrong_shape():
    # Model right below 2.5 keV, 40% too low above 5 keV: the global ratio would hide it, the bands do not
    model = _background() * np.where(ENERGY > 5, 0.6, 1.0)
    outcome = pid.FitOutcome((), model, 0.0, {}, {}, model)
    ratios = dict(pid.background_band_ratios(_background(), ENERGY, outcome, pid.PeakIDSettings().background_bands_keV))
    assert ratios[(1.0, 2.5)] == pytest.approx(1.0)
    assert ratios[(5.0, 100.0)] == pytest.approx(1 / 0.6)


def test_detector_artifact_bound():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Si": 4000})).astype(float)
    assert 0.002 < 4000 / y.sum() < 0.01
    kept = _identify(y, fixed=["Fe"])  # default bound: 0.2% of the counts
    assert kept.new_elements == ["Si"]
    removed = _identify(y, fixed=["Fe"], settings=pid.PeakIDSettings(artifact_min_fraction={"Si": 0.01}))
    assert removed.new_elements == []
    si = next(e for e in removed.elements if e.element == "Si")
    assert si.status == "pruned" and "detector artifact" in si.note


def test_strong_elements_have_residual_peaks():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Cu": 8000})).astype(float)
    result = _identify(y)
    for e in result.elements:
        if e.tier == "strong":
            assert e.residual_peak_ok and e.filtered_gain >= 100


def test_z_range_limits_candidates():
    rng = np.random.default_rng(RNG_SEED)
    y = rng.poisson(_background() + _family_spectrum({"Fe": 20000, "Cu": 8000})).astype(float)
    assert "Cu" in _identify(y, fixed=["Fe"]).new_elements
    # Cu (Z = 29) outside the allowed range: never proposed
    assert "Cu" not in _identify(y, fixed=["Fe"], settings=pid.PeakIDSettings(z_range=(5, 28))).new_elements
    assert "Pu" not in pid.PeakIDSettings().excluded_elements and pid.PeakIDSettings().z_range[1] == 83
