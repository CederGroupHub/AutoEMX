#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch quantification and analysis of X-ray spectra for a list of samples.

This script provides automated batch quantification and (optionally) clustering/statistical
analysis of acquired X-ray spectra for multiple samples. It is robust to missing files or
errors in individual samples, making it suitable for unattended batch processing.

Run this file directly to process the list of sample IDs with the defined configuration options.

Notes
-----
- Only the `sample_ID` is required if acquisition output is saved in the default directory;
  otherwise, specify `results_path`.
- Designed to continue processing even if some samples are missing or have errors.

Created on Tue Jul 29 13:18:16 2025

@author: Andrea
"""
from autoemxsp.runners.Batch_Quantify_and_Analyze import batch_quantify_and_analyze

# =============================================================================
# Examples
# =============================================================================
sample_IDs = [
    'Wulfenite_example',
    'K-412_NISTstd_example',
    # 'known_powder_mixture_example'
    ]

is_known_precursor_mixture = None # Loads value from config file, if unspecified. Otherwise set to True or False.

results_path = None # Relative path to folder where results are stored. Looks in default Results folder if left unspecified

# =============================================================================
# Options
# =============================================================================
max_analytical_error = 5 # w% Threhsold value of analytical error above which spectra are filtered out. Only used at the analysis stage, so it does not affect the quantification

min_bckgrnd_cnts = 5 # Minimum value of background counts that a reference peak (used for quantification) has to possess in order for measurement to be valid
    # Spectra not satisfying this are flagged (quant_flag = 8) and not quantified if interrupt_fits_bad_spectra = True. If False, they are still quantified, and filtered out later in the clustering stage
    # If too many spectra end up being flagged, decrease min_bckgrnd_cnts or increase the spectra target total counts
    # If you change min_bckgrnd_cnts, you can requantify the unquantified spectra only by setting quantify_only_unquantified_spectra = True


run_clustering_analysis = True # Whether to run the clustering analysis automatically after the quantification

num_CPU_cores = None # Number of cores used during fitting and quantification. If None, selects automatically half the available cores
quantify_only_unquantified_spectra = False # Set to True if running on Data.csv file that has already been quantified. Used to quantify discarded unqiantified spectra
interrupt_fits_bad_spectra = True # Interrupts the fit and quantification of spectra when it finds they will lead to large quantification errors. Used to speed up computations
use_project_specific_std_dict = None # If True, loads standards from project folder (i.e. results_dir) during quantification.

output_filename_suffix = '' # Suffix added to Analysis folder and Data.csv file

# =============================================================================
# Run
# =============================================================================
comp_analyzer = batch_quantify_and_analyze(
    sample_IDs=sample_IDs,
    quantification_method = 'PB',
    min_bckgrnd_cnts = min_bckgrnd_cnts,
    results_path=results_path,
    output_filename_suffix=output_filename_suffix,
    max_analytical_error=max_analytical_error,
    num_CPU_cores = num_CPU_cores,
    quantify_only_unquantified_spectra=quantify_only_unquantified_spectra,
    interrupt_fits_bad_spectra=interrupt_fits_bad_spectra,
    use_project_specific_std_dict = use_project_specific_std_dict,
    is_known_precursor_mixture = is_known_precursor_mixture,
    run_analysis=run_clustering_analysis,
)