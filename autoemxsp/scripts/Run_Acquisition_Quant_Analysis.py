#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated X-Ray Spectral Acquisition and Analysis

This script configures and runs automated collection and (optionally) quantification
of EDS/WDS spectra for powder samples using an electron microscope (EM) with
a specified substrate and calibration.

Typical usage:
    - Edit the 'samples' list to define your standards or unknowns
    - Adjust configuration parameters as needed
    - Run the script to perform spectrum collection and (optionally)
        quantification for one or multiple samples at a time
        
Created on Fri Jul 26 09:34:34 2024

@author: Andrea
"""

from autoemxsp.runners import batch_acquire_and_analyze

# =============================================================================
# General Configuration
# =============================================================================
microscope_ID = 'PhenomXL' # microscope ID, defines EDS spectral calibration and isntrument driver to use
microscope_type = 'SEM' # Currently only 'SEM' is supported
measurement_type = 'EDS' # Currently only 'EDS' is supported
measurement_mode = 'point' # Defines beam current and EDS detector calibration parameters
quantification_method = 'PB' # Currently only 'PB' is supported
beam_energy = 15  # keV
spectrum_lims = (14, 1100)  # indices to clip spectral data

working_distance = 5 #mm. Approximate WD at which sample is in focus. AutoEMXSp limits autofocus around this value to avoid gross msitakes in autofocus functions

use_instrument_background = False # If correctly set up, AutoEMXSp uses background function extracted from EDS software instead of extracting it during the fit
interrupt_fits_bad_spectra = True # Whether to interrupt the quantification of spectra expected to lead to gross quantification errors. Tested extensively. Speeds up quantification.

max_analytical_error_percent = 5 # Maximum analytical error to employ to filter out compositions during clustering. Can be modified later. Does not influence quantification
min_bckgrnd_cnts = 5 # Minimum number of counts under a reference peak necesary for a spectrum to be accepted for clustering. Can be modified later. Does not influence quantification
quant_flags_accepted = [0, -1] # Quantification flags accepted during clsutering (see docs). Can be modified later. Does not influence quantification

max_n_clusters = 6 # Max number of clusters. Can be modified later. Does not influence quantification
show_unused_comps_clust = True # Whether to show discarded compositions in clustering plot. Can be modified later. Does not influence quantification

output_filename_suffix = '' # Suffix to add to Analysis folder or Data.csv file. Normally left empty

# =============================================================================
# Sample Definitions
# =============================================================================
sample_type = 'powder' # Supported types: powder, bulk, powder_continuous, bulk_rough
sample_halfwidth = 3  # mm. Half-width of sample
sample_substrate_type = 'Ctape' # Supported types: Ctape, None
sample_substrate_shape = 'circle' # Supported types: square, circle
sample_substrate_width_mm = 12 # Al stub diameter, in mm

samples = [
    {'ID': 'Anorthite_mineral', 'els': ['Ca', 'Al', 'Si', 'O'], 'pos': (-37.5, -37.5), 'cnd': ['CaAl2Si2O8']},
]
# ID: Sample ID. All data will be saved in results_dir, in a folder named after 'ID'
# els: Elements to include during EDS spectral quantification
# pos: Position of center of sample. Does not require precise position of carbon tape if is_auto_substrate_detection = True
# cnd: Candidate phases that may be present in the sample. Only relevant during clustering analaysi, not for quantification.
#        Can be added or modified later when performing clustering analysis

results_dir = '' # Uses default directory if set to None. Otherwise creates a new folder at specified path.

# =============================================================================
# Acquisition and Quantification Options
# =============================================================================

is_manual_navigation = False # Whether to manually navigate the microscope to the desired frame to analyse

is_auto_substrate_detection = True # Whether to activate automated detection of substrate.
                                    # Only sample_substrate_type = 'Ctape' is currently supported for this option.
                                    # C tape must appear black on a brighter support stub, e.g. aluminum

auto_adjust_brightness_contrast = True # Use automatic adjustments of brightness and contrast
contrast = 4.3877  # Used if auto_adjust_brightness_contrast = False
brightness = 0.4504  # Used if auto_adjust_brightness_contrast = False

quantify_spectra = False

min_n_spectra = 50 # Min number of spectra after which AutoEMXSp checks for convergence. Only useful if quantify_spectra = True
max_n_spectra = 100 # Number of spectra collected if quantify_spectra = False. If quantify_spectra = True, this indicates the max number of spectra collected when convergence is not achieved.

target_Xsp_counts = 50000 # Target number of counts in each spectrum
max_XSp_acquisition_time = target_Xsp_counts / 10000 * 5 # Maximum acquisition time. Empyrically determined to stop when spectra are wrongfully acquired from C tape.

# Elements present in the substrate (may depend on target_Xsp_counts). These are ignored during quantification, unless present in the sample.
els_substrate = ['C', 'O', 'Al']  # N and F also detectable with >100k counts

# =============================================================================
# Powder sample options - Define parameters for particle segmentation and for determining EDS spot selection on particles
# =============================================================================
# To be used when sample_type = 'powder'
powder_meas_cfg_kwargs = dict(
    is_manual_particle_selection = False,
    is_known_powder_mixture_meas = False,
    max_n_par_per_frame=30,
    max_spectra_per_par=3,
    max_area_par=10000.0,
    min_area_par=10.0,
    par_mask_margin=1.0,
    xsp_spots_distance_um=1.0,
    par_brightness_thresh=100,
    par_xy_spots_thresh=100,
    par_feature_selection = 'random',
    par_spot_spacing = 'random'
)

# =============================================================================
# Bulk sample options - define acquisition grid
# =============================================================================
# To be used when sample_type = 'bulk' | 'powder_continuous' | 'bulk_rough'
bulk_meas_cfg_kwargs = dict(
    grid_spot_spacing_um = 100.0, # µm
    min_xsp_spots_distance_um = 5.0, # µm
    image_frame_width_um = 50, # µm
    randomize_frames = False,
    exclude_sample_margin = False
)

# =============================================================================
# Run
# =============================================================================
comp_analyzer = batch_acquire_and_analyze(
    samples=samples,
    microscope_ID=microscope_ID,
    microscope_type=microscope_type,
    measurement_type=measurement_type,
    measurement_mode=measurement_mode,
    quantification_method = quantification_method,
    sample_type=sample_type,
    sample_halfwidth=sample_halfwidth,
    sample_substrate_type=sample_substrate_type,
    sample_substrate_shape=sample_substrate_shape,
    sample_substrate_width_mm=sample_substrate_width_mm,
    working_distance = working_distance,
    beam_energy=beam_energy,
    spectrum_lims=spectrum_lims,
    use_instrument_background=use_instrument_background,
    interrupt_fits_bad_spectra=interrupt_fits_bad_spectra,
    max_analytical_error_percent=max_analytical_error_percent,
    min_bckgrnd_cnts=min_bckgrnd_cnts,
    quant_flags_accepted=quant_flags_accepted,
    max_n_clusters=max_n_clusters,
    show_unused_comps_clust=show_unused_comps_clust,
    is_manual_navigation=is_manual_navigation,
    is_auto_substrate_detection=is_auto_substrate_detection,
    auto_adjust_brightness_contrast=auto_adjust_brightness_contrast,
    contrast=contrast,
    brightness=brightness,
    quantify_spectra=quantify_spectra,
    min_n_spectra=min_n_spectra,
    max_n_spectra=max_n_spectra,
    target_Xsp_counts=target_Xsp_counts,
    max_XSp_acquisition_time=max_XSp_acquisition_time,
    els_substrate=els_substrate,
    powder_meas_cfg_kwargs=powder_meas_cfg_kwargs,
    bulk_meas_cfg_kwargs=bulk_meas_cfg_kwargs,
    output_filename_suffix=output_filename_suffix,
    development_mode=False,
    verbose=True,
    results_dir = results_dir
)