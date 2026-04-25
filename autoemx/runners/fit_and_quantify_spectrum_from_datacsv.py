#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitting and quantification of a single X-ray spectrum.

For spectrum-level analysis of fitting and quantification performance.

Import this module in your own code and call the
`fit_and_quantify_spectrum()` function, passing your desired 'sample_ID', 'spectrum_ID' and
options as arguments. This enables integration into larger workflows or pipelines.

Workflow:
    - Loads sample configurations from `Spectra_collection_info.json`
    - Loads acquired spectral data from `Data.csv`
    - Performs quantification (optionally only on unquantified spectra)
    - Optionally performs clustering/statistical analysis and saves results

Notes
-----
- Only the `sample_ID` and 'spectrum_ID' are required if acquisition output is saved in the default Results directory;
  otherwise, specify `results_path`.

Created on Tue Jul 29 13:18:16 2025

@author: Andrea
"""

import os
import warnings
import logging
from typing import Optional, List  

from autoemx.utils import (
    print_double_separator,
    get_sample_dir,
    load_configurations_from_json,
    extract_spectral_data
)
import autoemx.utils.constants as cnst
import autoemx.config.defaults as dflt
from autoemx.config import config_classes_dict, ExpStandardsConfig
from autoemx.config.schemas import ClusteringConfig
from autoemx.core.composition_analysis import EMXSp_Composition_Analyzer
from .fit_and_quantify_spectrum import fit_and_quantify_spectrum
from autoemx.config import load_sample_ledger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

__all__ = [
    "fit_and_quantify_spectrum_from_ledger",
    "fit_and_quantify_spectrum_fromDatacsv",
]

def fit_and_quantify_spectrum_from_ledger(
    sample_ID: str,
    spectrum_ID: int,
    els_sample: Optional[List[str]] = None,
    els_substrate: Optional[List[str]] = None,
    is_standard: bool = False,
    spectrum_lims: Optional[tuple] = None,
    results_path: Optional[str] = None,
    use_instrument_background: bool = dflt.use_instrument_background,
    quantify_plot: bool = True,
    plot_signal: bool = True,
    zoom_plot: bool = False,
    line_to_plot: str = '',
    fit_tol: float = 1e-4,
    is_particle: bool = True,
    max_undetectable_w_fr: float = 0,
    force_single_iteration: bool = False,
    interrupt_fits_bad_spectra: bool = False,
    standards_dict: Optional[dict] = None,
    print_results: bool = True,
    quant_verbose: bool = True,
    fitting_verbose: bool = True
):
    """
    Fit and (optionally) quantify a single spectrum.

    Parameters
    ----------
    sample_ID : str
        Sample identifier.
    spectrum_ID : int
        Value reported in 'Spectrum #' column in Data.csv.
    els_sample : list(str), optional
        List of elements in the sample. If the first entry is "" or None, the rest of the list is appended to the 
        list loaded from Comp_analysis_configs.json; otherwise, the provided list replaces it.
    els_substrate : list(str), optional
        List of substrate elements. If the first entry is "" or None, the rest of the list is appended to the 
        list loaded from Comp_analysis_configs.json; otherwise, the provided list replaces it.
    is_standard : bool
        Defines whether measurement is from an experimental standard (i.e., sample of known composition)
    results_path : str, optional
        Base directory where results are stored. Default: autoemx/Results
    use_instrument_background : bool, optional
        Whether to use instrument background if present. Default: False
    quantify_plot : bool, optional
        Whether to quantify the spectrum.
    plot_signal : bool, optional
        Whether to plot the fitted spectrum.
    zoom_plot : bool, optional
        Whether to zoom on a specific line.
    line_to_plot : str, optional
        Line to zoom on.
    fit_tol : float, optional
        scipy fit tolerance. Defines conditions of fit convergence
    is_particle : bool, optional
        If True, treats sample as particle (powder). Uses particle geometry fitting parameters
    max_undetectable_w_fr : float, optional
        Maximum allowed weight fraction for undetectable elements (default: 0). Total mass fraction of fitted
        elements is forced to be between [1-max_undetectable_w_fr, 1]
    force_single_iteration : bool, optional
        If True, quantification will be run for a single iteration only (default: False).
    interrupt_fits_bad_spectra : bool, optional
        If True, interrupt fitting if spectrum is detected to lead to poor quantification (default: False).
    print_results : bool, optional
        If True, prints all fitted parameters and their values (default: True).
    quant_verbose : bool, optional
        If True, prints quantification operations
    fitting_verbose : bool, optional
        If True, prints fitting operations
        
    Returns
    -------
    quantifier : XSp_Quantifier
        The quantifier object containing the results, fit parameters, and methods for further analysis and plotting.
    """
    if results_path is None:
        results_path = os.path.join(os.getcwd(), cnst.RESULTS_DIR)
        
    try:
        sample_dir = get_sample_dir(results_path, sample_ID)
    except Exception as e:
        logging.warning("Failed to get sample directory for %s: %s", sample_ID, e)
        return

    ledger_path = os.path.join(sample_dir, f"{cnst.LEDGER_FILENAME}{cnst.LEDGER_FILEEXT}")
    config_path_new = os.path.join(sample_dir, f"{cnst.CONFIG_FILENAME}.json")
    config_path_legacy = os.path.join(sample_dir, f"{cnst.ACQUISITION_INFO_FILENAME}.json")
    spectral_info_f_path = ledger_path if os.path.exists(ledger_path) else (
        config_path_new if os.path.exists(config_path_new) else config_path_legacy
    )
    data_filename = cnst.STDS_MEAS_FILENAME if is_standard else cnst.DATA_FILENAME
    data_path = os.path.join(sample_dir, f"{data_filename}.csv")
    
    print_double_separator()
    logging.info(f"Sample '{sample_ID}', spectrum {spectrum_ID}")
    
    try:
        if os.path.exists(ledger_path):
            ledger = load_sample_ledger(ledger_path)
            configs = {
                cnst.MICROSCOPE_CFG_KEY: ledger.configs.microscope_cfg,
                cnst.SAMPLE_CFG_KEY: ledger.configs.sample_cfg,
                cnst.MEASUREMENT_CFG_KEY: ledger.configs.measurement_cfg,
                cnst.SAMPLESUBSTRATE_CFG_KEY: ledger.configs.sample_substrate_cfg,
                cnst.PLOT_CFG_KEY: ledger.configs.plot_cfg,
            }
            if ledger.quantification_configs:
                active_quant_id = ledger.active_quant
                active_quant_config = next(
                    (
                        quant_config
                        for quant_config in ledger.quantification_configs
                        if quant_config.quantification_id == active_quant_id
                    ),
                    ledger.quantification_configs[-1],
                )
                configs[cnst.QUANTIFICATION_CFG_KEY] = config_classes_dict[cnst.QUANTIFICATION_CFG_KEY](
                    **active_quant_config.options
                )
                active_clustering_config = active_quant_config.get_active_clustering_config()
                if active_clustering_config is not None:
                    configs[cnst.CLUSTERING_CFG_KEY] = active_clustering_config
            else:
                configs[cnst.QUANTIFICATION_CFG_KEY] = config_classes_dict[cnst.QUANTIFICATION_CFG_KEY]()
            if ledger.configs.powder_meas_cfg is not None:
                configs[cnst.POWDER_MEASUREMENT_CFG_KEY] = ledger.configs.powder_meas_cfg
            if ledger.configs.bulk_meas_cfg is not None:
                configs[cnst.BULK_MEASUREMENT_CFG_KEY] = ledger.configs.bulk_meas_cfg
            metadata = {}
        else:
            configs, metadata = load_configurations_from_json(spectral_info_f_path, config_classes_dict)
    except FileNotFoundError:
        logging.warning(f"Could not find {spectral_info_f_path}. Skipping sample '{sample_ID}'.")
        return
    except Exception as e:
        logging.warning(f"Error loading {spectral_info_f_path}. Skipping sample '{sample_ID}': {e}")
        return

    # Retrieve configuration objects
    try:
        microscope_cfg      = configs[cnst.MICROSCOPE_CFG_KEY]
        sample_cfg          = configs[cnst.SAMPLE_CFG_KEY]
        measurement_cfg     = configs[cnst.MEASUREMENT_CFG_KEY]
        sample_substrate_cfg= configs[cnst.SAMPLESUBSTRATE_CFG_KEY]
        quant_cfg           = configs[cnst.QUANTIFICATION_CFG_KEY]
        clustering_cfg      = configs.get(cnst.CLUSTERING_CFG_KEY)
        powder_meas_cfg     = configs.get(cnst.POWDER_MEASUREMENT_CFG_KEY, None)  # Optional
        exp_stds_cfg     = configs.get(cnst.EXP_STD_MEASUREMENT_CFG_KEY, None)  # Optional
    except KeyError as e:
        logging.warning(f"Missing configuration '{e.args[0]}' in {spectral_info_f_path}. Skipping sample '{sample_ID}'.")
        return

    if clustering_cfg is None:
        clustering_cfg = ClusteringConfig()
    
    # Initialise analyzer so first-run legacy datasets are migrated to ledger,
    # then hydrate spectral arrays from ledger-managed sources.
    if powder_meas_cfg and powder_meas_cfg.is_known_powder_mixture_meas and not exp_stds_cfg:
        exp_stds_cfg = ExpStandardsConfig()

    comp_analyzer = EMXSp_Composition_Analyzer(
        microscope_cfg=microscope_cfg,
        sample_cfg=sample_cfg,
        measurement_cfg=measurement_cfg,
        sample_substrate_cfg=sample_substrate_cfg,
        quant_cfg=quant_cfg,
        initial_clustering_cfg=clustering_cfg,
        powder_meas_cfg=powder_meas_cfg,
        exp_stds_cfg=exp_stds_cfg,
        standards_dict=standards_dict,
        is_acquisition=False,
        verbose=True,
        results_dir=sample_dir,
    )

    # For legacy projects with Data.csv only, ensure migration path is available.
    if not os.path.exists(ledger_path):
        if not os.path.exists(data_path):
            logging.warning(f"Could not find {data_path}. Skipping sample '{sample_ID}'.")
            return
        try:
            extract_spectral_data(data_path)
        except Exception as e:
            logging.warning(f"Could not load spectral data for '{sample_ID}': {e}")
            return

    try:
        comp_analyzer._sync_in_memory_spectra_from_ledger()
    except Exception as e:
        logging.warning(f"Could not sync spectral data from ledger for '{sample_ID}': {e}")
        return

    spectral_data = comp_analyzer.spectral_data
    sp_coords = comp_analyzer.sp_coords
    stds_dict = comp_analyzer.XSp_std_dict

    # Resolve requested spectrum by Spectrum # (stored in sp_coords).
    sp_idx = None
    for i, sp_meta in enumerate(sp_coords):
        sp_id_val = sp_meta.get(cnst.SP_ID_DF_KEY)
        if sp_id_val is None:
            continue
        try:
            if int(float(sp_id_val)) == int(spectrum_ID):
                sp_idx = i
                break
        except (TypeError, ValueError):
            if str(sp_id_val) == str(spectrum_ID):
                sp_idx = i
                break

    if sp_idx is None:
        logging.warning(
            f"Spectrum ID {spectrum_ID} not found in ledger-backed data for sample '{sample_ID}'."
        )
        return

    try:
        spectrum = spectral_data[cnst.SPECTRUM_DF_KEY][sp_idx]
    except Exception as e:
        logging.warning(f"Spectrum data not found for spectrum ID {spectrum_ID} in sample '{sample_ID}': {e}")
        return

    # Background extraction
    background = None
    bkg_list = spectral_data.get(cnst.BACKGROUND_DF_KEY)
    
    if use_instrument_background:
        if (
            bkg_list is not None
            and isinstance(bkg_list, (list, tuple))
            and len(bkg_list) > sp_idx
            and bkg_list[sp_idx] is not None
        ):
            background = bkg_list[sp_idx]
        else:
            warnings.warn(
                "Instrument background not found or empty for this spectrum. "
                "Spectral background will be computed instead."
            )

    # Collection time
    if cnst.LIVE_TIME_DF_KEY in spectral_data and len(spectral_data[cnst.LIVE_TIME_DF_KEY]) > sp_idx:
        sp_collection_time = spectral_data[cnst.LIVE_TIME_DF_KEY][sp_idx]
    else:
        sp_collection_time = None

    # Calibration and configuration parameters
    try:
        beam_energy = measurement_cfg.beam_energy_keV
        emergence_angle = measurement_cfg.emergence_angle
        el_to_quantify = sample_cfg.elements
        offset = microscope_cfg.energy_zero
        scale = microscope_cfg.bin_width
    except Exception as e:
        logging.error(f"Error extracting calibration/configuration parameters: {e}")
        return
    
    # Sample elements
    if els_sample is None:
        els_sample = el_to_quantify
    elif (els_sample[0] == "" or els_sample[0] is None):
        els_sample = el_to_quantify + els_sample[1:]
            
    # Substrate elements
    if els_substrate is None:
        els_substrate = sample_substrate_cfg.elements
    elif (els_substrate[0] == "" or els_substrate[0] is None):
        els_substrate = sample_substrate_cfg.elements + els_substrate[1:]
        
    # Spectral limits
    if spectrum_lims is None:
        spectrum_lims = quant_cfg.spectrum_lims
    
    # Ensure spectrum_lims are integers for array slicing
    if spectrum_lims is not None:
        try:
            spectrum_lims = (int(spectrum_lims[0]), int(spectrum_lims[1]))
        except (TypeError, ValueError, IndexError):
            logging.error(f"Invalid spectrum_lims: {spectrum_lims}")
            return
    else:
        logging.error("spectrum_lims could not be determined")
        return
            
    quantifier = fit_and_quantify_spectrum(
        spectrum_vals = spectrum,
        spectrum_lims = spectrum_lims,
        microscope_ID = microscope_cfg.ID,
        meas_type = measurement_cfg.type,
        meas_mode = measurement_cfg.mode,
        det_ch_offset=offset,
        det_ch_width=scale,
        beam_energy = beam_energy,
        emergence_angle = emergence_angle,
        sp_collection_time = sp_collection_time,
        sample_ID = sample_ID,
        els_sample = els_sample,
        els_substrate = els_substrate,
        background_vals=background,
        fit_tol = fit_tol,
        is_particle = is_particle,
        quantify_plot = quantify_plot,
        max_undetectable_w_fr = max_undetectable_w_fr,
        force_single_iteration = force_single_iteration,
        interrupt_fits_bad_spectra = interrupt_fits_bad_spectra,
        standards_dict = stds_dict,
        plot_signal = plot_signal,
        plot_title = f"{sample_ID}_#{spectrum_ID}",
        zoom_plot = zoom_plot,
        line_to_plot = line_to_plot,
        print_results = print_results,
        quant_verbose = quant_verbose,
        fitting_verbose = fitting_verbose
    )

    return quantifier


def fit_and_quantify_spectrum_fromDatacsv(*args, **kwargs):
    """Deprecated alias for backward compatibility; use fit_and_quantify_spectrum_from_ledger."""
    warnings.warn(
        "fit_and_quantify_spectrum_fromDatacsv is deprecated; "
        "use fit_and_quantify_spectrum_from_ledger instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_and_quantify_spectrum_from_ledger(*args, **kwargs)