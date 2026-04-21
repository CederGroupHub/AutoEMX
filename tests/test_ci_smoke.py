#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CI-safe smoke tests for core package workflows.

These tests intentionally avoid hardware-dependent paths and interactive plotting.
"""

import os
from pathlib import Path

from autoemxsp.runners.Fit_and_Quantify_Spectrum_fromDatacsv import (
    fit_and_quantify_spectrum_fromDatacsv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "autoemxsp" / "Results"


# Force a non-interactive backend for CI runners.
os.environ.setdefault("MPLBACKEND", "Agg")


def test_single_spectrum_quantification_smoke():
    quantifier = fit_and_quantify_spectrum_fromDatacsv(
        sample_ID="Wulfenite_example",
        spectrum_ID=4,
        results_path=str(RESULTS_PATH),
        is_standard=False,
        quantify_plot=False,
        plot_signal=False,
        zoom_plot=False,
        fit_tol=1e-4,
        is_particle=True,
        max_undetectable_w_fr=0,
        force_single_iteration=False,
        interrupt_fits_bad_spectra=False,
        print_results=False,
        quant_verbose=False,
        fitting_verbose=False,
    )

    assert quantifier is not None
    assert hasattr(quantifier, "fit_res")
    assert quantifier.fit_res is not None
