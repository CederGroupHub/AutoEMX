#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch quantification and analysis of X-ray spectra for known precursor mixes.

Used to assess the extent of intermixing of precurors prior to a reaction.
See example at:
    L. N. Walters et al., Synthetic Accessibility and Sodium Ion Conductivity of the Na 8– x A x P 2 O 9 (NAP)
    High-Temperature Sodium Superionic Conductor Framework, Chem. Mater. 37, 6807 (2025).

This script provides automated batch quantification and (optionally) clustering/statistical
analysis of acquired X-ray spectra for multiple powder mixes. It is robust to missing files or
errors in individual samples, making it suitable for unattended batch processing.

Created on Tue Jul 29 13:18:16 2025

@author: Andrea
"""

from autoemxsp.runners import batch_quantify_and_analyze

# =============================================================================
# Samples
# =============================================================================  
is_known_precursor_mixture = True

sample_IDs = ['known_powder_mixture_example']

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
    is_known_precursor_mixture = is_known_precursor_mixture,
    run_analysis=run_clustering_analysis,
)