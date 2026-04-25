#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch quantification and analysis of X-ray spectra for a list of samples.

This script provides automated batch quantification and clustering/statistical
analysis of acquired X-ray spectra for multiple samples. It is robust to missing files or
errors in individual samples, making it suitable for unattended batch processing.

Run this file directly to process the list of sample IDs with the defined configuration options.

Notes
-----
- Only the `sample_ID` is required if acquisition output is saved in the default directory;
  otherwise, specify `results_path`.
- Clustering/statistical analysis runs by default (runner default: `run_analysis=True`).
- Designed to continue processing even if some samples are missing or have errors.

Created on Tue Jul 29 13:18:16 2025

@author: Andrea
"""
import os

# =============================================================================
# Examples
# =============================================================================
sample_IDs = [
    'Wulfenite_example',
    # 'K-412_NISTstd_example'
    ]

results_path = os.path.join(os.path.dirname(__file__), 'Results')
# =============================================================================
# Options
# =============================================================================
max_analytical_error = 5 # w%
min_bckgrnd_cnts = 5

num_CPU_cores = None # If None, selects automatically half the available cores
interrupt_fits_bad_spectra = True # Interrupts the fit and quantification of spectra when it finds they will lead to large quantification errors. Used to speed up computations. If False, previously interrupted spectra are re-quantified without interruption.

output_filename_suffix = ''

# =============================================================================
# Run
# =============================================================================
from autoemx.runners.Batch_Quantify_and_Analyze import batch_quantify_and_analyze

comp_analyzer = batch_quantify_and_analyze(
    sample_IDs=sample_IDs,
    quantification_method = 'PB',
    min_bckgrnd_cnts = min_bckgrnd_cnts,
    results_path=results_path,
    output_filename_suffix=output_filename_suffix,
    max_analytical_error=max_analytical_error,
    num_CPU_cores = num_CPU_cores,
    interrupt_fits_bad_spectra=interrupt_fits_bad_spectra,
)