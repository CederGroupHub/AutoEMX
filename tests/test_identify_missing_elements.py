#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the identification of missing elements (elements missing from the fit) in quantification."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ci_paths import INPUTS_DIR, WULFENITE_MINI_ID
import autoemx.utils.constants as cnst
from autoemx.config.runtime_configs import QuantificationOptionsConfig
from autoemx.config.schema_models.ledger import SampleLedger
from autoemx.config.schema_models.quantification import QuantificationConfig
from autoemx.core.quantifier import XSp_Quantifier
from autoemx.runners.batch_quantify_and_analyze import batch_quantify_and_analyze

REQUIRED_OPTIONS = {
    "method": "PB",
    "spectrum_lims": [14, 1100],
    "fit_tolerance": 1e-3,
    "use_instrument_background": False,
}


# =============================================================================
# Configuration
# =============================================================================
def test_legacy_options_default_to_no_identify_missing_elements():
    legacy = QuantificationConfig(quantification_id=0, sample_elements=["Mo"], options=REQUIRED_OPTIONS)
    explicit = QuantificationConfig(
        quantification_id=0, sample_elements=["Mo"],
        options={**REQUIRED_OPTIONS, "identify_missing_elements": False, "identification_seed_spectra": 10},
    )
    assert legacy.options["identify_missing_elements"] is False
    assert legacy.fingerprint() == explicit.fingerprint()


def test_enabling_identify_missing_elements_changes_fingerprint():
    base = QuantificationConfig(quantification_id=0, sample_elements=["Mo"], options=REQUIRED_OPTIONS)
    checked = QuantificationConfig(
        quantification_id=0, sample_elements=["Mo"], options={**REQUIRED_OPTIONS, "identify_missing_elements": True},
    )
    assert base.fingerprint() != checked.fingerprint()


def test_options_config_validates_seed_spectra():
    assert QuantificationOptionsConfig().identify_missing_elements is True
    assert QuantificationOptionsConfig(identify_missing_elements=True, identification_seed_spectra=5).identification_seed_spectra == 5
    with pytest.raises(ValidationError):
        QuantificationOptionsConfig(identification_seed_spectra=-1)


# =============================================================================
# Quantifier (one spectrum of the mini wulfenite ledger, PbMoO4)
# =============================================================================
def _wulfenite_quantifier(els_sample, identify_missing_elements=True):
    sample_dir = Path(INPUTS_DIR) / WULFENITE_MINI_ID
    ledger = SampleLedger.from_json_file(str(sample_dir / "ledger.json"))
    entry = ledger.spectra[0]
    spectrum = np.asarray(ledger._load_counts_from_pointer_file(sample_dir / entry.spectrum_relpath), dtype=float)
    mc, me = ledger.configs.microscope_cfg, ledger.configs.measurement_cfg
    return XSp_Quantifier(
        spectrum_vals=spectrum, spectrum_lims=(14, 1100), microscope_ID=mc.ID, meas_type=me.type, meas_mode=me.mode,
        det_ch_offset=mc.energy_zero, det_ch_width=mc.bin_width, beam_e=me.beam_energy_keV,
        emergence_angle=me.emergence_angle, els_sample=els_sample, els_substrate=["C", "O", "Al"],
        is_particle=True, sp_collection_time=entry.live_acquisition_time, identify_missing_elements=identify_missing_elements,
    )


def test_has_standard():
    quantifier = _wulfenite_quantifier(["Pb", "Mo", "O"], identify_missing_elements=False)
    assert quantifier._has_standard("Mo")
    assert not quantifier._has_standard("Pm")


def test_missing_element_is_added_and_quantified():
    quantifier = _wulfenite_quantifier(["Pb", "O"])
    quant_result, _, _ = quantifier.quantify_spectrum(print_result=False, interrupt_fits_bad_spectra=False)
    assert quantifier.added_elements == ["Mo"]
    assert quant_result is not None and "Mo" in quant_result[cnst.COMP_AT_FR_KEY]
    record = quantifier.export_quantification_result(quantification_id=0, quant_result=quant_result)
    assert record.diagnostics.added_elements == ["Mo"]
    assert "Mo" in record.composition_atomic_fractions


def test_nothing_added_with_complete_element_list():
    quantifier = _wulfenite_quantifier(["Pb", "Mo", "O"])
    quantifier.quantify_spectrum(print_result=False, interrupt_fits_bad_spectra=False)
    assert quantifier.added_elements == []
    assert quantifier.added_unquantified_elements == []


# =============================================================================
# Batch quantification and reporting
# =============================================================================
def test_batch_quantification_reports_added_elements(results_path):
    analyzers = batch_quantify_and_analyze(
        sample_IDs=[WULFENITE_MINI_ID],
        els_sample=["Pb", "O"],
        quantification_method="PB",
        results_path=results_path,
        run_analysis=False,
        interrupt_fits_bad_spectra=False,
        force_requantification=True,
        identify_missing_elements=True,
    )
    analyzer = analyzers[0]
    summary = analyzer.element_identification_summary()
    assert "Mo" in summary and len(summary["Mo"]["quantified"]) == 2
    compositions = pd.read_csv(os.path.join(analyzer.analysis_dir, "Compositions.csv"))
    assert "Added elements" in compositions.columns
    assert all("Mo" in str(v) for v in compositions["Added elements"])
    with open(os.path.join(analyzer.analysis_dir, "Element_identification_report.txt"), encoding="utf-8") as file:
        assert "Mo: added and quantified in 2 spectra" in file.read()
