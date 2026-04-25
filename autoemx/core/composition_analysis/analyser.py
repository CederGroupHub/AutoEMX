#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMXSp_Composition_Analyzer

Main class for automated compositional analysis of electron microscopy X-ray spectroscopy (EMXSp) data.

Can be run from run_acquisition_quant_analysis.py

Features:
- Structured configuration for microscope, sample, measurement, and analysis parameters.
- Automated acquisition and quantification of X-ray spectra at electron microscope.
- Filtering and clustering of compositional data.
- Phase identification, mixture analysis, and comprehensive results export.
- Utilities for plotting, saving, and reporting analysis results.

Example Usage
-------------
    # Create analyzer instance
    >>> analyzer = EMXSp_Composition_Analyzer(
            microscope_cfg=microscope_cfg,
            sample_cfg=sample_cfg,
            measurement_cfg=measurement_cfg,
            sample_substrate_cfg=sample_substrate_cfg,
            quant_cfg=quant_cfg,
            initial_clustering_cfg=initial_clustering_cfg,
            powder_meas_cfg=powder_meas_cfg,
            plot_cfg=plot_cfg,
            is_acquisition=True,
            development_mode=False,
            output_filename_suffix='',
            verbose=True,
        )

    # Acquire and quantify spectra, and analyse compositions
    >>> analyzer.run_collection_and_quantification(quantify=True)

    # Alternatively, acquire only, then quantify:
    >>> analyzer.run_collection_and_quantification(quantify=False)
    >>> quantify and analyse on another machine using Run_Quantification.py


@author: Andrea
Created on Mon Jul 22 17:43:35 2024
"""

# Standard library imports
import os
import json
import time
import shutil
import itertools
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Any, Optional, Tuple, List, Dict, Iterable, Union
from joblib import Parallel, delayed

#TODO remove in future versions
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="cvxpy.*",
)

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns
from pymatgen.core.composition import Composition
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import cvxpy as cp
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Project-specific imports
from autoemx.core.quantifier import XSp_Quantifier, Quant_Corrections
from autoemx.core.em_runtime.controller import EM_Controller
from autoemx.core.em_runtime.sample_finder import EM_Sample_Finder
import autoemx.calibrations as calibs
import autoemx.utils.constants as cnst
import autoemx.config.defaults as dflt
import autoemx._custom_plotting as custom_plotting
from autoemx.utils import (
    print_single_separator,
    print_double_separator,
    to_latex_formula,
    make_unique_path,
    weight_to_atomic_fr
)
from autoemx.config import (
    MicroscopeConfig,
    SampleConfig,
    MeasurementConfig,
    SampleSubstrateConfig,
    QuantificationOptionsConfig,
    PowderMeasurementConfig,
    BulkMeasurementConfig,
    ExpStandardsConfig,
    PlotConfig,
)
from autoemx.config.schemas import (
    AcquisitionDetails,
    ClusteringConfig as LedgerClusteringConfig,
    Coordinate2D,
    LedgerConfigs,
    QuantificationConfig,
    QuantificationDiagnostics,
    QuantificationResult,
    SampleLedger,
    SpotCoordinates,
    SpectrumEntry,
)
from .clustering import ClusteringModule
from autoemx.utils.legacy.legacy_backfill import backfill_spectra_from_data_csv, load_ledger_configs_from_legacy_json
from autoemx.utils.legacy.legacy_ledger_loader import (
    build_legacy_import_quantification_config,
    load_legacy_acquisition_details_by_spectrum_id,
    load_legacy_quantification_results_by_spectrum_id,
)
from .plotting import PlottingModule
from .reference_matching import ReferenceMatchingModule
from autoemx.utils.legacy.spectrum_pointer_writer import load_vendor_msa_template_lines, write_spectrum_pointer_file
from .standards import StandardsModule

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#%% EMXSp_Composition_Analyzer class
class EMXSp_Composition_Analyzer:
    """
    Main class for electron microscopy X-ray spectroscopy (EMXSp) composition analysis.

    This class orchestrates the acquisition, quantification, clustering, and plotting
    of X-ray spectra and composition data, using structured configuration objects for
    all instrument and analysis settings.

    Parameters
    ----------
    microscope_cfg : MicroscopeConfig
        Configuration for the microscope hardware.
    sample_cfg : SampleConfig
        Configuration for the sample.
    measurement_cfg : MeasurementConfig
        Configuration for the measurement/acquisition.
    sample_substrate_cfg : SampleSubstrateConfig
        Configuration for the sample substrate.
    quant_cfg : QuantificationOptionsConfig
        Configuration for spectrum fitting and quantification.
    initial_clustering_cfg : autoemx.config.schemas.ClusteringConfig
        Initial clustering settings used when no active quantification config is yet available in the ledger.
    powder_meas_cfg : PowderMeasurementConfig
        Configuration for powder measurement.
    bulk_meas_cfg : BulkMeasurementConfig
        Configuration for measurements of bulk or bulk-like samples.
    exp_stds_cfg : ExpStandardsConfig
        Configuration for measurements of experimental standards.
    plot_cfg : PlotConfig
        Configuration for plotting.
    is_acquisition : bool, optional
        If True, indicates class is being used for automated acquisition (default: False).
    standards_dict : dict, optional
        Dictionary of reference PB values from experimental standards. Default : None.
        If None, dictionary of standards is loaded from the XSp_calibs/Your_Microscope_ID directory.
        Provide standards_dict only when providing different standards from those normally used for quantification.
    development_mode : bool, optional
        If True, enables development/debug features (default: False).
    output_filename_suffix : str, optional
        String to append to saved filenames (default: '').
    verbose : bool, optional
        If True, enables verbose output (default: True).
    results_dir : Optional[str], optional
        Directory to save results (default: None). If None, uses default directory, created inside package folder
            - Results, for sample analysis
            - Std_measurements, for experimental standard measurements

    Attributes
    ----------
    TO COMPLETE
    """
    #TODO
    @staticmethod
    def _coerce_optional_finite_float(value: Any) -> Optional[float]:
        """Return a finite float or None when the input is missing/non-finite."""
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return numeric_value

    def __init__(
        self,
        microscope_cfg: MicroscopeConfig,
        sample_cfg: SampleConfig,
        measurement_cfg: MeasurementConfig,
        sample_substrate_cfg: SampleSubstrateConfig,
        quant_cfg: Optional[QuantificationOptionsConfig] = QuantificationOptionsConfig(),
        initial_clustering_cfg: Optional[Any] = None,
        powder_meas_cfg: Optional[PowderMeasurementConfig] = PowderMeasurementConfig(),
        bulk_meas_cfg: Optional[BulkMeasurementConfig] = BulkMeasurementConfig(),
        exp_stds_cfg: Optional[ExpStandardsConfig] = ExpStandardsConfig(),
        plot_cfg: Optional[PlotConfig] = PlotConfig(),
        is_acquisition: bool = False,
        standards_dict: Optional[dict] = None,
        development_mode: bool = False,
        output_filename_suffix: str = '',
        verbose: bool = True,
        results_dir: Optional[str] = None,
    ):
        """
        Initialize the EMXSp_Composition_Analyzer with all configuration objects.

        See class docstring for parameter documentation.
        """
        # --- Record process time
        self.start_process_time = time.time()
        if verbose:
            print_double_separator()
            print(f"Starting compositional analysis of sample {sample_cfg.ID}")
            
            
        # --- Define use of class instance
        self.is_acquisition = is_acquisition
        is_XSp_measurement = measurement_cfg.type != measurement_cfg.PARTICLE_STATS_MEAS_TYPE_KEY
        self.development_mode = development_mode
        
        
        # --- System characteristics
        self.microscope_cfg = microscope_cfg
        
        if is_XSp_measurement:
            # Load microscope calibrations for this instrument and mode
            calibs.load_microscope_calibrations(microscope_cfg.ID, measurement_cfg.mode, load_detector_channel_params=is_acquisition)
            if not measurement_cfg.emergence_angle:
                measurement_cfg.emergence_angle = calibs.emergence_angle # Fixed by instrument geometry
        
        
        # --- Measurement configurations
        self.measurement_cfg = measurement_cfg
        self.powder_meas_cfg = powder_meas_cfg
        self.bulk_meas_cfg = bulk_meas_cfg
        self.exp_stds_cfg = exp_stds_cfg
        
        if is_XSp_measurement:
            if is_acquisition:
                # Loaded latest detector calibration values
                meas_modes_calibs = calibs.detector_channel_params
                energy_zero = meas_modes_calibs[measurement_cfg.mode][cnst.OFFSET_KEY]
                bin_width = meas_modes_calibs[measurement_cfg.mode][cnst.SCALE_KEY]
                beam_current = meas_modes_calibs[measurement_cfg.mode][cnst.BEAM_CURRENT_KEY]
                # Store, because needed to call XS_Quantifier
                self.microscope_cfg.energy_zero = energy_zero
                self.microscope_cfg.bin_width = bin_width
                if not measurement_cfg.beam_current:
                    self.measurement_cfg.beam_current = beam_current
                    
                # --- Type checking ---
                for var_name, var_value in [
                    ('energy_zero', energy_zero),
                    ('bin_width', bin_width),
                    ('beam_current', beam_current)
                ]:
                    if not isinstance(var_value, float):
                        raise TypeError(f"{var_name} must be a float, got {type(var_value).__name__}: {var_value}")
            else:
                energy_zero = microscope_cfg.energy_zero
                bin_width = microscope_cfg.bin_width
                beam_current = measurement_cfg.beam_current
                
            self.det_ch_offset = energy_zero
            self.det_ch_width = bin_width

            # Max and min number of EDS spectra to be collected
            self.min_n_spectra = measurement_cfg.min_n_spectra
            if measurement_cfg.max_n_spectra < self.min_n_spectra:
                self.max_n_spectra = self.min_n_spectra
            else:
                self.max_n_spectra = measurement_cfg.max_n_spectra
            
            
        # --- Sample characteristics
        self.sample_cfg = sample_cfg         # Elements possibly present in the sample
        self.sample_substrate_cfg = sample_substrate_cfg
        if is_XSp_measurement:
            # Elements possibly present in the sample
            self.all_els_sample = list(dict.fromkeys(sample_cfg.elements)) #remove any eventual duplicate, keeping original order
            # Detectable elements possibly present in the sample 
            self.detectable_els_sample = [el for el in self.all_els_sample if el not in calibs.undetectable_els]
            # Elements present in the substrate, which have to be subtracted if not present in the sample
            self.all_els_substrate = list(dict.fromkeys(sample_substrate_cfg.elements)) #remove any eventual duplicate, keeping original order
            detectable_els_substrate = [el for el in self.all_els_substrate if el not in calibs.undetectable_els] # remove undetectable elements
            self.detectable_els_substrate = [el for el in detectable_els_substrate if el not in self.detectable_els_sample] #remove any eventual duplicate
            self._apply_geom_factors = True if sample_cfg.is_surface_rough else False
        
        
        # --- Fitting and Quantification
        self.quant_cfg = quant_cfg
        self.standards_dict = standards_dict

        if initial_clustering_cfg is None:
            self.clustering_cfg = LedgerClusteringConfig(features=cnst.AT_FR_CL_FEAT)
        elif hasattr(initial_clustering_cfg, "model_dump"):
            clustering_payload = initial_clustering_cfg.model_dump(mode="json")
            clustering_payload.pop("clustering_id", None)
            self.clustering_cfg = LedgerClusteringConfig.model_validate(clustering_payload)
        elif isinstance(initial_clustering_cfg, dict):
            clustering_payload = dict(initial_clustering_cfg)
            clustering_payload.pop("clustering_id", None)
            self.clustering_cfg = LedgerClusteringConfig.model_validate(clustering_payload)
        else:
            raise TypeError(
                "initial_clustering_cfg must be None, a mapping, or a model with model_dump()"
            )

        if is_XSp_measurement:
            # Set EDS detector channels to include in the quantification
            self.sp_start, self.sp_end = (
                int(round(quant_cfg.spectrum_lims[0])),
                int(round(quant_cfg.spectrum_lims[1])),
            )
            # Compute values of energies corresponding to detector channels
            if energy_zero and bin_width:
                self.energy_vals = np.array([energy_zero + bin_width * i for i in range(self.sp_start, self.sp_end)])
            elif is_acquisition and is_XSp_measurement:
                raise ValueError("Missing detector calibration values.\n Please add detector calibration file at {calibs.calibration_files_dir}")
            # Set a threshold value below which counts are considered to be too low
            # Used to filter "bad" spectra out from clustering analysis. All spectra having less counts than this threshold are filtered out
            # Used also to avoid fitting spectra with excessive absorption, which inevitably lead to large quantification errors
            if self.clustering_cfg.min_bckgrnd_cnts is None:
                min_bckgrnd_cnts = measurement_cfg.target_acquisition_counts / (2 * 10**4)  # empirical value
                self.clustering_cfg.min_bckgrnd_cnts = int(round(min_bckgrnd_cnts))

        # --- Clustering
        if is_XSp_measurement:
            self.ref_formulae = list(dict.fromkeys(self.clustering_cfg.ref_formulae)) # Remove duplicates
            self._calc_reference_phases_df() # Calculate dataframe with reference compositions, and any possible analyitical error deriving from undetectable elements
        
        
        # --- Plotting
        self.plot_cfg = plot_cfg
        
        
        # --- Output
        # Create a new directory if acquiring
        if is_acquisition:
            if results_dir is None:
                if self.exp_stds_cfg.is_exp_std_measurement:
                    results_folder = cnst.STDS_DIR
                else:
                    results_folder = cnst.RESULTS_DIR
                results_dir = make_unique_path(os.path.join(os.getcwd(), results_folder), sample_cfg.ID)
                os.makedirs(results_dir)
            else:
                # Reuse the provided directory (resume or continuation scenario).
                os.makedirs(results_dir, exist_ok=True)

        self.sample_result_dir = results_dir
        self.output_filename_suffix = output_filename_suffix
        self.verbose = verbose

        if is_XSp_measurement:
            # --- Variable initialization
            self.XSp_std_dict = None
            self.sp_coords = [] # List containing particle number + relative coordinates on the image to retrieve exact position where spectra were collected
            self.particle_cntr = -1 # Counter to save particle number
            
            # Initialise lists containing spectral data and comments that will be saved with the quantification data
            self.spectral_data = {key : [] for key in cnst.LIST_SPECTRAL_DATA_KEYS}
            
            # List containing the quantification result records for each collected spectrum.
            # spectra_quant_records[i] is a QuantificationResult (or None when not yet quantified).
            # All composition/error/fit data needed for analysis is read directly from these records.
            self.spectra_quant_records = []
            self.current_quant_config: Optional[QuantificationConfig] = None
            self.current_quantification_id: Optional[int] = None
            
        
        # --- Save configurations
        # Save spectrum collection info when class is used to collect spectra
        if is_acquisition:
            self._save_experimental_config(is_XSp_measurement)
        
        
        # --- Initialisations
        # Initialise microscope and XSp analyser
        if is_acquisition:
            if microscope_cfg.type == 'SEM':
                self._initialise_SEM()
            
            if is_XSp_measurement:
                self._initialise_Xsp_analyzer()
                self._initialise_acquisition_ledger()
        

    #%% Instrument initializations
    # =============================================================================
    def _initialise_SEM(self) -> None:
        """
        Initialize the SEM (Scanning Electron Microscope) and related analysis tools.
    
        Sets up the instrument controller, directories, and, if applicable,
        initializes the particle finder for automated powder sample analysis.
        For circular sample substrates, it automatically detects the C-tape position.
    
        Raises
        ------
        FileNotFoundError
            If the sample result directory does not exist and cannot be created.
        NotImplementedError
            If the sample type is not 'powder'.
        """
        # Determine collection and detection modes based on user-configured settings
        is_manual_navigation = self.measurement_cfg.is_manual_navigation
        is_auto_substrate_detection = self.sample_substrate_cfg.auto_detection
    
        # If using automated collection with a circular sample substrate, detect C-tape and update sample coordinates (center, radius)
        if not is_manual_navigation and is_auto_substrate_detection:
            sample_finder = EM_Sample_Finder(
                microscope_ID=self.microscope_cfg.ID,
                center_pos=self.sample_cfg.center_pos,
                sample_half_width_mm=self.sample_cfg.half_width_mm,
                substrate_width_mm=self.sample_substrate_cfg.stub_w_mm,
                results_dir=self.sample_result_dir,
                verbose=self.verbose
            )
            if self.sample_substrate_cfg.type == cnst.CTAPE_SUBSTRATE_TYPE:
                Ctape_coords = sample_finder.detect_Ctape()
                if Ctape_coords:
                    center_pos, C_tape_r = Ctape_coords
                    # Update detected center position and half-width
                    self.sample_cfg.center_pos = center_pos
                    self.sample_cfg.half_width_mm = C_tape_r
            else:
                warnings.warn(f"Automatic detection is only implemented for {cnst.ALLOWED_AUTO_DETECTION_TYPES}")
        
        # Set up image directory for this sample
        EM_images_dir = os.path.join(self.sample_result_dir, cnst.IMAGES_DIR)
        if not os.path.exists(EM_images_dir):
            try:
                os.makedirs(EM_images_dir)
            except Exception as e:
                raise FileNotFoundError(f"Could not create results directory: {EM_images_dir}") from e
    
        # Initialise instrument controller
        self.EM_controller = EM_Controller(
            self.microscope_cfg,
            self.sample_cfg,
            self.measurement_cfg,
            self.sample_substrate_cfg,
            self.powder_meas_cfg,
            self.bulk_meas_cfg,
            results_dir=EM_images_dir,
            verbose=self.verbose
        )
        self.EM_controller.initialise_SEM()
        self.EM_controller.initialise_sample_navigator(exclude_sample_margin=True)
        
        # Update employed working distance
        self.measurement_cfg.working_distance = self.EM_controller.measurement_cfg.working_distance
            
            
    def _initialise_Xsp_analyzer(self):
        """
        Initialize the X-ray spectroscopy analyzer according to the measurement configuration.
    
        If the measurement type is 'EDS', this initializes the EDS (Energy Dispersive X-ray Spectroscopy)
        analyzer via the associated EM_controller. For any other measurement type, a NotImplementedError
        is raised.
    
        Raises
        ------
        NotImplementedError
            If the measurement type is not 'EDS'.
        """
        # Only EDS is supported at present
        if self.measurement_cfg.type == 'EDS':
            self.EM_controller.initialise_XS_analyzer()
        elif self.measurement_cfg.type not in self.measurement_cfg.ALLOWED_TYPES:
            raise NotImplementedError(
                f"X-ray spectroscopy analyzer initialization for measurement type '{self.measurement_cfg.type}' is not currently implemented."
            )

    def _initialise_acquisition_ledger(self) -> None:
        """Create the sample ledger with configs at acquisition start, before any spectra are collected.

        Does nothing if a ledger already exists (resume scenario).
        The ledger is created with an empty spectra list; entries are populated from
        the written .msa pointer files when quantification is later launched.
        """
        ledger_path = self._get_ledger_path()
        if os.path.exists(ledger_path):
            return
        spectra_dir = self._get_spectra_dir()
        os.makedirs(spectra_dir, exist_ok=True)
        ledger = SampleLedger(
            sample_id=self.sample_cfg.ID,
            sample_path=os.path.abspath(self.sample_result_dir),
            configs=self._build_ledger_configs(),
            spectra=[],
            quantification_configs=[],
            active_quant=None,
        )
        ledger.to_json_file(ledger_path)

    #%% Other initializations
    # =============================================================================
    def _make_analysis_dir(self) -> None:
        """
        Create a deterministic directory for saving analysis results.

        The directory name is based on the active quantification and clustering config ids:
        ``analysis_Quant{quantification_id}_Clust{clustering_id}``.
        If the directory already exists (same active config pair), it is reused as-is.
        Existing files are preserved unless overwritten by newly generated outputs.
        A different active config pair results in a different directory name.
    
        The resulting directory path is stored in `self.analysis_dir`.
    
        Raises
        ------
        RuntimeError
            If active quantification/clustering ids cannot be resolved.
        OSError
            If directory creation fails.
        """
        quantification_id, clustering_id = self._resolve_active_analysis_config_ids()
        base_name = f"analysis_Quant{quantification_id}_Clust{clustering_id}"
        analysis_dir = os.path.join(self.sample_result_dir, base_name)

        try:
            os.makedirs(analysis_dir, exist_ok=True)
        except Exception as e:
            raise OSError(f"Could not create analysis directory '{analysis_dir}': {e}") from e

        self.analysis_dir = analysis_dir


    def _save_analysis_config_summary(self) -> None:
        """Write a human-readable plain-text summary of the quantification and
        clustering configurations used to produce this analysis folder.

        The file is written to ``self.analysis_dir/Analysis_config_summary.txt``.
        It is always overwritten so it reflects the configs that generated the
        current files in the folder.
        """
        lines: list[str] = []

        # Quantification section
        q  = self.quant_cfg             # QuantificationOptionsConfig
        qc = self.current_quant_config  # QuantificationConfig from ledger (may be None)
        if qc is None and self.sample_result_dir is not None:
            try:
                _ledger = self._load_or_create_ledger()
                qc = self._get_active_quantification_config(_ledger)
            except Exception:
                pass

        # Prefer options from the active QuantificationConfig stored in ledger,
        # because those are the exact scientific inputs used for this analysis.
        active_options = qc.options if (qc is not None and isinstance(qc.options, dict)) else {}
        method_used = str(active_options.get("method", q.method))
        fit_tolerance_used = float(active_options.get("fit_tolerance", q.fit_tolerance))
        use_instr_background_used = bool(
            active_options.get("use_instrument_background", q.use_instrument_background)
        )
        use_proj_std_dict_used = bool(
            active_options.get("use_project_specific_std_dict", q.use_project_specific_std_dict)
        )
        spectrum_lims_used = active_options.get("spectrum_lims", q.spectrum_lims)
        if isinstance(spectrum_lims_used, (list, tuple)) and len(spectrum_lims_used) == 2:
            sp_low = int(float(spectrum_lims_used[0]))
            sp_high = int(float(spectrum_lims_used[1]))
        else:
            sp_low = int(float(q.spectrum_lims[0]))
            sp_high = int(float(q.spectrum_lims[1]))

        lines.append("=" * 60)
        lines.append("QUANTIFICATION CONFIG")
        lines.append("=" * 60)
        if qc is not None:
            lines.append(f"  Quantification ID    : {qc.quantification_id}")
            if qc.label:
                lines.append(f"  Label                : {qc.label}")
            if qc.sample_elements:
                lines.append(f"  Sample elements      : {', '.join(qc.sample_elements)}")
            if qc.substrate_elements:
                lines.append(f"  Substrate elements   : {', '.join(qc.substrate_elements)}")
        lines.append(f"  Method               : {method_used}")
        lines.append(f"  Spectrum limits      : {sp_low} - {sp_high} channels")
        lines.append(f"  Fit tolerance        : {fit_tolerance_used:.2e}")
        lines.append(f"  Instrument background: {'yes' if use_instr_background_used else 'no'}")
        lines.append(f"  Project std dict     : {'yes' if use_proj_std_dict_used else 'no'}")
        if qc is not None and qc.reference_lines_by_element:
            sample_elements = set(qc.sample_elements or [])
            reference_lines_by_element = {
                el: line
                for el, line in qc.reference_lines_by_element.items()
                if el in sample_elements
            }
            ref_lines_str = ', '.join(
                f'{el} -> {ln}' for el, ln in sorted(reference_lines_by_element.items())
            )
            if ref_lines_str:
                lines.append(f"  Reference lines      : {ref_lines_str}")

        # Clustering section
        cc = self.clustering_cfg

        lines.append("")
        lines.append("=" * 60)
        lines.append("CLUSTERING CONFIG")
        lines.append("=" * 60)
        if hasattr(cc, "clustering_id"):
            lines.append(f"  Clustering ID        : {cc.clustering_id}")
        lines.append(f"  Method               : {cc.method}")
        lines.append(f"  Features             : {cc.features}")
        if cc.k_forced is not None:
            lines.append(f"  k (forced)           : {cc.k_forced}")
        else:
            lines.append(f"  k finding method     : {cc.k_finding_method}")
            lines.append(f"  Max k                : {cc.max_k}")
        if getattr(cc, "k_resolved", None) is not None:
            lines.append(f"  k (resolved)         : {cc.k_resolved}")
        lines.append(f"  Max analytical error : {cc.max_analytical_error_percent} w%")
        lines.append(
            f"  Min background counts: {cc.min_bckgrnd_cnts if cc.min_bckgrnd_cnts is not None else 'disabled'}"
        )
        lines.append(f"  Accepted quant flags : {cc.quant_flags_accepted}")
        lines.append(f"  Matrix decomposition : {'yes' if cc.do_matrix_decomposition else 'no'}")
        if cc.ref_formulae:
            lines.append("  Reference formulae   :")
            for formula in cc.ref_formulae:
                lines.append(f"    - {formula}")
        else:
            lines.append("  Reference formulae   : (none)")
        lines.append("")

        summary_path = os.path.join(
            self.analysis_dir,
            cnst.ANALYSIS_CONFIG_SUMMARY_FILENAME + ".txt",
        )
        try:
            with open(summary_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        except OSError as e:
            logging.warning(f"Could not write analysis config summary: {e}")


    def _resolve_active_analysis_config_ids(self) -> Tuple[int, int]:
        """Resolve active quantification and clustering config ids for analysis folder naming."""
        quant_config = self.current_quant_config

        if quant_config is None:
            ledger = self._load_or_create_ledger()
            quant_config = self._get_active_quantification_config(ledger)

        if quant_config is None:
            raise RuntimeError("No active quantification config is available to name analysis directory")

        active_clustering_config = quant_config.get_active_clustering_config()
        if active_clustering_config is None:
            raise RuntimeError("No active clustering config is available to name analysis directory")

        return int(quant_config.quantification_id), int(active_clustering_config.clustering_id)


    @staticmethod
    def _clear_directory_contents(dir_path: str) -> None:
        """Remove all files/subdirectories from a directory while keeping the directory itself."""
        try:
            for entry in os.scandir(dir_path):
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)
        except Exception as e:
            raise OSError(f"Could not clear existing analysis directory '{dir_path}': {e}") from e
    
    
    def _initialise_std_dict(self) -> None:
        """
        Initialise the dictionary of X-ray standards for quantification.
    
        This method determines how the `XSp_std_dict` attribute is initialised
        based on the sample configuration and measurement type:
    
        - If the measurement is of a known powder mixture, the standards dictionary
          is compiled from reference data using `_compile_standards_from_references()`.
    
        - Otherwise, the standards dictionary is expected to be loaded directly
          within the `XSp_Quantifier` and is set to `None` here.
    
        Returns
        -------
        None
            This method modifies the `self.XSp_std_dict` attribute in place.
        """
        is_known_mixture = getattr(self.powder_meas_cfg, "is_known_powder_mixture_meas", False)
        
        if is_known_mixture:
            self.XSp_std_dict = self._compile_standards_from_references()
        elif self.quant_cfg.use_project_specific_std_dict:
            std_dict_all_modes, _ = self._load_xsp_standards()
            std_dict = std_dict_all_modes[self.measurement_cfg.mode]
            self.XSp_std_dict = std_dict
        else:
            # Standards dictionary will be loaded directly within the `XSp_Quantifier`
            self.XSp_std_dict = None


    def _calc_reference_phases_df(self) -> None:
        """
        Calculate the compositions of candidate phases and store them in a pd.DataFrame.
    
        For each reference formula in `self.ref_formulae`, this method:
          - Computes the composition using pymatgen's Composition class.
          - Computes either mass or atomic fractions, depending on clustering configuration.
          - Accounts for undetectable elements and calculates the maximum analytical error due to their presence.
          - Stores the resulting phase compositions in `self.ref_phases_df` and the weights in `self.ref_weights_in_mixture`.
    
        If no reference formulae are provided, the function exits without error.
    
        Warnings
        --------
        Issues a warning if a formula cannot be parsed or if no detectable elements are found in a formula.
    
        Raises
        ------
        ValueError
            If an unknown clustering feature set is specified.
        """
        import warnings
    
        undetectable_an_err = 0
        ref_phases = []
        ref_weights_in_mixture = []
        
        # Check if self.ref_formulae is set to None
        if not self.ref_formulae:
            # No reference formulae provided; nothing to do
            self.ref_phases_df = pd.DataFrame(columns=self.all_els_sample)
            self.ref_weights_in_mixture = []
            self.undetectable_an_er = 0
            return
        
        valid_formulae = []
        valid_compositions = set()  # store normalized Composition keys, to check for duplicates

        for formula in self.ref_formulae:
            # Use pymatgen class Composition
            try:
                comp = Composition(formula)
            except Exception as e:
                warnings.warn(f"Invalid chemical formula '{formula}': {e}")
                continue
            
            # Normalize composition to a string key to check duplicates
            comp_key = comp.reduced_formula  # or str(comp) if you want exact
            if comp_key in valid_compositions:
                continue  # skip duplicate compositions
            
            valid_compositions.add(comp_key)
            valid_formulae.append(formula)
            
            # Get mass fractions as dictionary el: w_fr
            try:
                w_fr_dict = comp.as_weight_dict()
            except AttributeError:
                w_fr_dict = comp.to_weight_dict
    
            # Check for detectable elements at the beginning
            detectable_in_formula = [el for el in self.detectable_els_sample if el in w_fr_dict]
            if not detectable_in_formula:
                warnings.warn(f"No detectable elements found in formula '{formula}'.")
                continue
    
            # Calculate analytical error due to undetectable elements
            for el, w_fr in w_fr_dict.items():
                if el in calibs.undetectable_els:
                    undetectable_an_err = max(undetectable_an_err, w_fr)
    
            if self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
                # Mass fractions are not normalised, so a negative analytical error is possible when undetectable elements are present
                # Calculate reference dictionary considering only quantified elements (e.g. Li is ignored)
                phase = {el: w_fr_dict.get(el, 0) for el in self.detectable_els_sample}
                # Store weight of reference in an eventual mixture, which is simply equal to the compound molar weight
                ref_weights_in_mixture.append(comp.weight)
    
            elif self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
                # Atomic fractions are normalised, so for the purpose of candidate phases we should calculate it normalising the
                # mass fractions, after discarding the undetectable elements
                detectable_w_frs = {el: w_fr for el, w_fr in w_fr_dict.items() if el in self.detectable_els_sample}
                # Transform to Composition class
                detectable_comp = comp.from_weight_dict(detectable_w_frs)
                # Get dictionary of el : at_fr
                phase = detectable_comp.fractional_composition.as_dict()
                # Store weight of reference in an eventual mixture, which is equal to the number of atoms in reference formula, without undetectable elements
                ref_weight = sum(at_n for el, at_n in comp.get_el_amt_dict().items() if el in self.detectable_els_sample)
                ref_weights_in_mixture.append(ref_weight)
            else:
                raise ValueError(f"Unknown clustering feature set: {self.clustering_cfg.features}")
    
            ref_phases.append(phase)
        
        # Copy all valid formulae back onto self.ref_formulae attribute
        self.ref_formulae = valid_formulae
        
        # Convert to pd.DataFrame and store it
        ref_phases_df = pd.DataFrame(ref_phases, columns=self.all_els_sample).fillna(0)
        self.ref_phases_df = ref_phases_df
    
        # Store values of reference weights used to calculate molar fractions from mixtures
        self.ref_weights_in_mixture = ref_weights_in_mixture
    
        # Calculate negative analytical error accepted to compensate for elements undetectable by EDS (H, He, Li, Be)
        self.undetectable_an_er = undetectable_an_err
        
        
    #%% Single spectrum operations
    # =============================================================================            
    def _acquire_spectrum(
        self,
        x: float,
        y: float,
        spectrum_id: str,
        msa_file_path: Optional[str] = None,
    ) -> Tuple[float, int]:
        """
        Acquire an X-ray spectrum at the specified stage position and store the results.
    
        Parameters
        ----------
        x, y : float
            X, Y coordinates for the spectrum acquisition.
            Coordinate System
            ----------------
            The coordinates are expressed in a normalized, aspect-ratio-correct system centered at the image center:
    
                - The origin (0, 0) is at the image center.
                - The x-axis is horizontal, increasing to the right, ranging from -0.5 (left) to +0.5 (right).
                - The y-axis is vertical, increasing downward, and scaled by the aspect ratio (height/width):
                    * Top edge:    y = -0.5 × (height / width)
                    * Bottom edge: y = +0.5 × (height / width)
                
                |        (-0.5, -0.5*height/width)         (0.5, -0.5*height/width)
                |                       +-------------------------+
                |                       |                         |
                |                       |                         |
                |                       |           +(0,0)        |-----> +x
                |                       |                         |
                |                       |                         |
                v  +y                   +-------------------------+
                        (-0.5,  0.5*height/width)         (0.5, 0.5*height/width)
    
            This ensures the coordinate system is always centered and aspect-ratio-correct, regardless of image size.
    
        Returns
        -------
        collection_time : float
            Real acquisition time used, read from the persisted pointer file when possible.
        total_counts : int
            Total counts in the acquired spectrum, derived from persisted data.
    
        Notes
        -----
        - Results are appended to self.spectral_data using the keys defined in `cnst`.
        """
        # Acquire at the instrument and rely on persisted pointer files as source of truth.
        background_elements = None
        if self.quant_cfg.use_instrument_background:
            background_elements = list(getattr(self, "detectable_els_sample", []) or [])

        spectrum_data, background_data = self.EM_controller.acquire_XS_spot_spectrum(
            x, y,
            self.measurement_cfg.max_acquisition_time,
            self.measurement_cfg.target_acquisition_counts,
            elements=background_elements,
            msa_file_path=msa_file_path,
        )

        counts_arr = np.asarray(spectrum_data, dtype=float)
        real_time = None
        if msa_file_path:
            pointer_path = Path(msa_file_path)
            loaded_counts = SampleLedger._load_counts_from_pointer_file(pointer_path)
            counts_arr = np.asarray(loaded_counts, dtype=float)
            real_time = self._load_realtime_from_pointer_file(pointer_path)
        if real_time is None:
            real_time = 1.0

        if self.quant_cfg.use_instrument_background and background_data is not None:
            self._write_manufacturer_background_vector(
                spectrum_id=spectrum_id,
                background_vals=list(map(float, background_data)),
            )
        elif self.quant_cfg.use_instrument_background and background_data is None:
            warnings.warn(
                "Instrument background retrieval failed during acquisition; "
                "falling back to automatic background subtraction for the "
                "remaining spectra in this run."
            )
            self.quant_cfg.use_instrument_background = False
    
        # Store results in the spectral_data dictionary
        self.spectral_data[cnst.SPECTRUM_DF_KEY].append(counts_arr)
        # Background vectors are file-backed sidecars and hydrated before quantification.
        self.spectral_data[cnst.BACKGROUND_DF_KEY].append(None)
        self.spectral_data[cnst.REAL_TIME_DF_KEY].append(real_time)
        self.spectral_data[cnst.LIVE_TIME_DF_KEY].append(real_time)
    
        return real_time, int(round(float(np.sum(counts_arr))))
    
    
    def _fit_exp_std_spectrum(
        self,
        spectrum: Iterable,
        background: Optional[Iterable] = None,
        sp_collection_time: float = None,
        els_w_frs: Optional[Dict[str,float]] = None,
        sp_id: str = '',
        verbose: bool = True
    ) -> Optional[Dict]:
        """
        Quantify a single X-ray spectrum.
    
        This method checks if the spectrum is valid for fitting, runs the quantification,
        flags the result as necessary, and appends comments and quantification flags to
        the spectral data attributes.
    
        Parameters
        ----------
        spectrum : Iterable
            The spectrum data to be quantified.
        background : Iterable, optional
            The background data associated with the spectrum.
        sp_collection_time : float, optional
            The collection time for the spectrum.
        sp_id: str, optional
            The spectrum ID, used as label for printing
        verbose : bool, optional
            If True, enables verbose output (default: True).
    
        Returns
        -------
        fit_result : Dict or None
            Dictionary returned by XSp_Quantifier, containing calculated composition in atomic fractions and
            analytical error, or None if the spectrum is not suitable for quantification or fitting fails.
    
        Notes
        -----
        - Filtering flags are appended through function _check_fit_quant_validity().
        """
        if verbose:
            if sp_id != '':
                sp_id_str = " #" + sp_id
            else:
                sp_id_str = '...'
            print_single_separator()
            print('Fitting spectrum' + sp_id_str)
            start_quant_time = time.time()
                            
        # Check if spectrum is worth fitting
        is_sp_valid_for_fitting, quant_flag, comment = self._is_spectrum_valid_for_fitting(spectrum, background)
        if not is_sp_valid_for_fitting:
            return None, quant_flag, comment
        
        # Initialize class to quantify spectrum
        quantifier = XSp_Quantifier(
            spectrum_vals=spectrum,
            spectrum_lims=(self.sp_start, self.sp_end),
            microscope_ID=self.microscope_cfg.ID,
            meas_type=self.measurement_cfg.type,
            meas_mode=self.measurement_cfg.mode,
            det_ch_offset=self.det_ch_offset,
            det_ch_width=self.det_ch_width,
            beam_e=self.measurement_cfg.beam_energy_keV,
            emergence_angle=self.measurement_cfg.emergence_angle,
            energy_vals=None,
            background_vals=background,
            els_sample=self.all_els_sample,
            els_substrate=self.detectable_els_substrate,
            els_w_fr=self.exp_stds_cfg.w_frs,
            is_particle=self._apply_geom_factors,
            sp_collection_time=sp_collection_time,
            max_undetectable_w_fr=self.undetectable_an_er,
            fit_tol=self.quant_cfg.fit_tolerance,
            standards_dict=self.XSp_std_dict,
            verbose=False,
            fitting_verbose=False
        )
        
        try:
            bad_quant_flag = quantifier.initialize_and_fit_spectrum(print_results=self.verbose)
            is_fit_valid = True
            min_bckgrnd_ref_lines = quantifier._get_min_bckgrnd_cnts_ref_quant_lines()
        except Exception as e:
            is_fit_valid = False
            print(f"{type(e).__name__}: {e}")
            traceback.print_exc()
            quant_flag, comment = self._check_fit_quant_validity(is_fit_valid, None, None, None)
            return None, quant_flag, comment
        
        fit_results_dict, are_all_ref_peaks_present = self._assemble_fit_info(quantifier)
        
        if are_all_ref_peaks_present:
            quant_flag, comment = self._check_fit_quant_validity(is_fit_valid, bad_quant_flag, quantifier, min_bckgrnd_ref_lines)
        else:
            comment = "Reference peak missing"
            quant_flag = 10
        
        if verbose:
            fit_time = time.time() - start_quant_time
            print(f"Fitting took {fit_time:.2f} s")
    
        return fit_results_dict, quant_flag, comment
    
    
    def _assemble_fit_info(self, quantifier):
        are_all_ref_peaks_present = True
        
        # Get fit result data to retrieve PB ratio 
        fit_data = quantifier.fitted_peaks_info
        
        reduced_chi_squared = quantifier.fit_result.redchi
        r_squared = 1 - quantifier.fit_result.residual.var() / np.var(quantifier.spectrum_vals)
        
        # Initialise variables
        PB_ratios_d = {} # Dictionary used to store the PB ratios of each line fitted in the spectrum
        
        # Store PB ratios from fitted peaks
        el_lines = [el_line for el_line in fit_data.keys() if 'esc' not in el_line and 'pileup' not in el_line]
        for el_line in el_lines:
            el, line = el_line.split('_')[:2]
            
            if el not in self.detectable_els_sample:
                continue # Do not store PB ratios for substrate elements
            
            meas_PB_ratio = fit_data[el_line][cnst.PB_RATIO_KEY]

            # Assign a nan value if PB ratio is too low, to later filter only the significant peaks
            if meas_PB_ratio < self.exp_stds_cfg.min_acceptable_PB_ratio:
                meas_PB_ratio = np.nan
            
            # Store PB ratio information
            if line in quantifier.xray_quant_ref_lines:
                # Store PB-ratio value, only for reference peaks
                PB_ratios_d[el_line] = meas_PB_ratio
                
                # Store theoretical energy values for fitted peaks
                self._th_peak_energies[el_line] = fit_data[el_line][cnst.PEAK_TH_ENERGY_KEY] 
                
                other_xray_ref_lines = [l for l in quantifier.xray_quant_ref_lines if l != line]

                # Elements of the standard must be properly fitted, and possess a background with enough counts
                if el in self.detectable_els_sample and all(el + '_' + l not in fit_data.keys() for l in other_xray_ref_lines):
                    # Check if peak is present
                    if not meas_PB_ratio > 0:
                        are_all_ref_peaks_present = False
                        # Reference peak not present
                        if self.verbose:
                            print(f"{el_line} reference peak missing.")
                    
        # Create dictionary of fit results
        fit_results_dict = {**PB_ratios_d, cnst.R_SQ_KEY : r_squared, cnst.REDCHI_SQ_KEY : reduced_chi_squared}

        # Append to list of results
        return fit_results_dict, are_all_ref_peaks_present
        
    
    def _fit_quantify_spectrum(
        self,
        spectrum: Iterable,
        background: Optional[Iterable] = None,
        sp_collection_time: float = None,
        sp_id: str = '',
        spectrum_index: Optional[int] = None,
        interrupt_fits_bad_spectra: bool = True,
        verbose: bool = True
    ) -> Optional[Dict]:
        """
        Quantify a single X-ray spectrum.
    
        This method checks if the spectrum is valid for fitting, runs the quantification,
        flags the result as necessary, and appends comments and quantification flags to
        the spectral data attributes.
    
        Parameters
        ----------
        spectrum : Iterable
            The spectrum data to be quantified.
        background : Iterable, optional
            The background data associated with the spectrum.
        sp_collection_time : float, optional
            The collection time for the spectrum.
        sp_id: str, optional
            The spectrum ID, used as label for printing
        verbose : bool, optional
            If True, enables verbose output (default: True).
    
        Returns
        -------
        quant_result : Dict or None
            Dictionary returned by XSp_Quantifier, containing calculated composition in atomic fractions and
            analytical error, or None if the spectrum is not suitable for quantification or fitting fails.
    
        Notes
        -----
        - Filtering flags are appended through function _check_fit_quant_validity().
        """
        if verbose:
            if sp_id != '':
                sp_id_str = " #" + sp_id
            else:
                sp_id_str = '...'
            print_single_separator()
            print('Quantifying spectrum' + sp_id_str)
            start_quant_time = time.time()
                            
        # Check if spectrum is worth fitting
        is_sp_valid_for_fitting, quant_flag, comment = self._is_spectrum_valid_for_fitting(spectrum, background)
        if not is_sp_valid_for_fitting:
            quant_record = QuantificationResult(
                quantification_id=self.current_quantification_id,
                quant_flag=quant_flag,
                comment=comment,
                diagnostics=QuantificationDiagnostics(
                    converged=False,
                    interrupted=True,
                ),
            )
            return None, quant_record, quant_flag, comment

        # Initialize class to quantify spectrum
        quantifier = XSp_Quantifier(
            spectrum_vals=spectrum,
            spectrum_lims=(self.sp_start, self.sp_end),
            microscope_ID=self.microscope_cfg.ID,
            meas_type=self.measurement_cfg.type,
            meas_mode=self.measurement_cfg.mode,
            det_ch_offset=self.det_ch_offset,
            det_ch_width=self.det_ch_width,
            beam_e=self.measurement_cfg.beam_energy_keV,
            emergence_angle=self.measurement_cfg.emergence_angle,
            energy_vals=None,
            background_vals=background,
            els_sample=self.all_els_sample,
            els_substrate=self.detectable_els_substrate,
            els_w_fr=self.sample_cfg.w_frs,
            is_particle=self._apply_geom_factors,
            sp_collection_time=sp_collection_time,
            max_undetectable_w_fr=self.undetectable_an_er,
            fit_tol=self.quant_cfg.fit_tolerance,
            standards_dict=self.XSp_std_dict,
            verbose=False,
            fitting_verbose=False
        )
        
        try:
            # Returns dictionary containing calculated composition in atomic fractions + analytical error
            quant_result, min_bckgrnd_ref_lines, bad_quant_flag = quantifier.quantify_spectrum(
                print_result=False,
                interrupt_fits_bad_spectra=interrupt_fits_bad_spectra
            )
            is_quant_fit_valid = True if quant_result is not None else False
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            is_quant_fit_valid = False
            quant_flag, comment = self._check_fit_quant_validity(is_quant_fit_valid, None, None, None)
            quant_record = quantifier.export_quantification_result(
                quantification_id=self.current_quantification_id,
                quant_result=None,
                quant_flag=quant_flag,
                comment=comment,
            )
            return None, quant_record, quant_flag, comment
        else:
            quant_flag, comment = self._check_fit_quant_validity(is_quant_fit_valid, bad_quant_flag, quantifier, min_bckgrnd_ref_lines)
            quant_record = quantifier.export_quantification_result(
                quantification_id=self.current_quantification_id,
                quant_result=quant_result,
                quant_flag=quant_flag,
                comment=comment,
            )
        
        if verbose and quant_result:
            quantification_time = time.time() - start_quant_time
            for el in quant_result[cnst.COMP_AT_FR_KEY].keys():
                print(f"{el} at%: {quant_result[cnst.COMP_AT_FR_KEY][el]*100:.2f}%")
            print(f"An. er.: {quant_result[cnst.AN_ER_KEY]*100:.2f}%")
            print(f"Quantification took {quantification_time:.2f} s")
    
        return quant_result, quant_record, quant_flag, comment


    def _get_ledger_path(self) -> str:
        """Return the ledger path for the current sample result directory."""
        return os.path.join(self.sample_result_dir, cnst.LEDGER_FILENAME + cnst.LEDGER_FILEEXT)


    def _resolve_or_create_spectrum_pointer(
        self,
        spectrum_id: str,
        spectrum_vals: List[float],
        *,
        live_time: Optional[float] = None,
        real_time: Optional[float] = None,
    ) -> str:
        """Return a relative spectrum pointer path and create the spectrum file when missing.

        This is used by ledger reconstruction/update code paths (including legacy
        Data.csv backfill). If no vendor template file is available in the sample
        directory, the spectrum is written with the minimal EMSA fallback format.
        """
        spectrum_relpath = self._build_spectrum_relpath(spectrum_id)
        spectrum_pointer_abs_path = os.path.join(self.sample_result_dir, spectrum_relpath)

        if os.path.exists(spectrum_pointer_abs_path):
            return spectrum_relpath

        if not hasattr(self, "_vendor_msa_template_lines"):
            setattr(
                self,
                "_vendor_msa_template_lines",
                load_vendor_msa_template_lines(self.sample_result_dir, cnst.MSA_SP_FILENAME),
            )

        write_spectrum_pointer_file(
            spectrum_pointer_abs_path,
            spectrum_vals,
            self.energy_vals,
            template_lines=getattr(self, "_vendor_msa_template_lines"),
            live_time=live_time,
            real_time=real_time,
        )
        return spectrum_relpath


    def _build_spectrum_relpath(self, spectrum_id: str) -> str:
        """Build the relative path for one raw spectrum pointer file."""
        filename = f"{cnst.SPECTRUM_FILENAME_PREFIX}{spectrum_id}{dflt.RAW_SPECTRUM_EXT}"
        return os.path.join(cnst.SPECTRA_DIR, filename)


    def _build_background_relpath(self, spectrum_id: str) -> str:
        """Build relative path for one manufacturer background vector file."""
        filename = (
            f"{cnst.SPECTRUM_FILENAME_PREFIX}{spectrum_id}"
            f"{cnst.SPECTRUM_MAN_BACKGROUND_SUFFIX}{cnst.VECTOR_FILEEXT}"
        )
        return os.path.join(cnst.SPECTRA_DIR, filename)


    def _write_manufacturer_background_vector(self, spectrum_id: str, background_vals: Optional[List[float]]) -> Optional[str]:
        """Persist manufacturer background counts as a companion vector file when enabled."""
        if background_vals is None or not self.quant_cfg.use_instrument_background:
            return None

        background_relpath = self._build_background_relpath(spectrum_id)
        background_abs_path = os.path.join(self.sample_result_dir, background_relpath)
        os.makedirs(os.path.dirname(background_abs_path), exist_ok=True)
        np.save(background_abs_path, np.asarray(list(map(float, background_vals)), dtype=float))
        return background_relpath


    @staticmethod
    def _load_background_vector_from_file(background_path: Path) -> Optional[np.ndarray]:
        """Load an instrument background vector from a sidecar file when available."""
        if not background_path.exists() or background_path.suffix.lower() != cnst.VECTOR_FILEEXT:
            return None
        try:
            background_vals = np.load(background_path, allow_pickle=False)
        except Exception:
            return None
        if background_vals is None:
            return None
        return np.asarray(background_vals, dtype=float)


    @staticmethod
    def _load_realtime_from_pointer_file(pointer_path: Path) -> Optional[float]:
        """Read REALTIME from an EMSA-like header when available."""
        if pointer_path.suffix.lower() not in {".msa", ".msg"}:
            return None

        try:
            with pointer_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line.startswith("#") or ":" not in line:
                        continue
                    if line.upper().startswith("#SPECTRUM"):
                        break
                    key, value = line[1:].split(":", maxsplit=1)
                    key_norm = key.strip().replace("_", "").replace(" ", "").upper()
                    if key_norm == "REALTIME":
                        return float(value.strip())
        except Exception:
            return None

        return None


    def _load_existing_ledger(self) -> Optional[SampleLedger]:
        """Load an existing ledger if present and valid."""
        ledger_path = self._get_ledger_path()
        if not os.path.exists(ledger_path):
            return None
        try:
            return SampleLedger.from_json_file(ledger_path)
        except Exception:
            return None


    def _get_spectra_dir(self) -> str:
        """Return the absolute path to the spectra pointer directory."""
        return os.path.join(self.sample_result_dir, cnst.SPECTRA_DIR)


    def _list_pointer_files_in_spectra_dir(self) -> List[Path]:
        """List pointer files currently present in the spectra directory."""
        spectra_dir = Path(self._get_spectra_dir())
        if not spectra_dir.exists():
            return []

        allowed_ext = {".msa", ".msg", ".json"}
        files = []
        for path in spectra_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in allowed_ext:
                continue
            stem = path.stem
            if not stem.startswith(cnst.SPECTRUM_FILENAME_PREFIX):
                continue
            if stem.endswith(cnst.SPECTRUM_MAN_BACKGROUND_SUFFIX):
                continue
            files.append(path)

        def sort_key(path: Path) -> Tuple[int, Union[int, str], str]:
            stem = path.stem
            spectrum_id = stem[len(cnst.SPECTRUM_FILENAME_PREFIX):] if stem.startswith(cnst.SPECTRUM_FILENAME_PREFIX) else stem
            if spectrum_id.isdigit():
                return (0, int(spectrum_id), path.name)
            return (1, spectrum_id.lower(), path.name)

        return sorted(files, key=sort_key)


    def _populate_spectra_dir_from_data_csv(self) -> int:
        """Backfill spectra pointer files from Data.csv when legacy datasets lack external spectra files."""
        data_csv_path = os.path.join(self.sample_result_dir, cnst.DATA_FILENAME + cnst.DATA_FILEEXT)
        if not os.path.exists(data_csv_path):
            return 0

        n_written = backfill_spectra_from_data_csv(
            data_csv_path,
            self._resolve_or_create_spectrum_pointer,
            spectrum_key=cnst.SPECTRUM_DF_KEY,
            spectrum_id_key=cnst.SP_ID_DF_KEY,
            live_time_key=cnst.LIVE_TIME_DF_KEY,
            real_time_key=cnst.REAL_TIME_DF_KEY,
            background_key=cnst.BACKGROUND_DF_KEY,
            write_background_pointer=(
                self._write_manufacturer_background_vector
                if self.quant_cfg.use_instrument_background
                else None
            ),
        )

        if n_written > 0 and not getattr(self, "_legacy_backfill_warned", False):
            warnings.warn(
                "Deprecation warning: legacy Data.csv compatibility path was used to reconstruct spectra files. "
                "Please reanalyse all old samples so a ledger is created; all new AutoEMX versions will read that ledger.",
                UserWarning,
            )
            self._legacy_backfill_warned = True

        return n_written


    @staticmethod
    def _parse_optional_int(value: Any) -> Optional[int]:
        if value is None or pd.isna(value) or value == "":
            return None
        try:
            return int(float(value))
        except Exception:
            return None


    @staticmethod
    def _parse_optional_float(value: Any) -> Optional[float]:
        if value is None or pd.isna(value) or value == "":
            return None
        try:
            return float(value)
        except Exception:
            return None


    def _build_spot_coordinates(
        self,
        machine_x: Any,
        machine_y: Any,
        pixel_x: Any = None,
        pixel_y: Any = None,
    ) -> Optional[SpotCoordinates]:
        machine_coords = None
        pixel_coords = None

        x_machine = self._parse_optional_float(machine_x)
        y_machine = self._parse_optional_float(machine_y)
        if x_machine is not None and y_machine is not None:
            machine_coords = Coordinate2D(x=x_machine, y=y_machine)

        x_pixel = self._parse_optional_float(pixel_x)
        y_pixel = self._parse_optional_float(pixel_y)
        if x_pixel is not None and y_pixel is not None:
            pixel_coords = (int(round(x_pixel)), int(round(y_pixel)))

        if machine_coords is None and pixel_coords is None:
            return None

        return SpotCoordinates(
            machine_coordinates=machine_coords,
            pixel_coordinates=pixel_coords,
        )


    def _build_spectrum_entry_from_pointer_file(
        self,
        pointer_file: Path,
        existing_results: Optional[List[QuantificationResult]] = None,
        acquisition_details_by_id: Optional[Dict[str, AcquisitionDetails]] = None,
        quantification_results_by_id: Optional[Dict[str, List[QuantificationResult]]] = None,
    ) -> SpectrumEntry:
        """Build a SpectrumEntry by inspecting one file under sample_path/spectra."""
        sample_root = Path(self.sample_result_dir).resolve()
        pointer_abs = pointer_file.resolve()
        pointer_rel = str(pointer_abs.relative_to(sample_root))
        stem = pointer_file.stem
        if stem.startswith(cnst.SPECTRUM_FILENAME_PREFIX):
            spectrum_id = stem[len(cnst.SPECTRUM_FILENAME_PREFIX):]
        else:
            spectrum_id = stem

        try:
            counts = SampleLedger._load_counts_from_pointer_file(pointer_abs)
            total_counts = int(round(float(np.sum(counts))))
        except Exception:
            total_counts = 0

        acquisition_details = AcquisitionDetails(frame_id=None, particle_id=None, spot_coordinates=None)
        if acquisition_details_by_id is not None:
            acquisition_details = acquisition_details_by_id.get(spectrum_id, acquisition_details)

        realtime_from_header = self._coerce_optional_finite_float(self._load_realtime_from_pointer_file(pointer_abs))
        background_relpath = None
        candidate_background = Path(
            self.sample_result_dir,
            self._build_background_relpath(spectrum_id),
        )
        if candidate_background.exists():
            background_relpath = str(candidate_background.resolve().relative_to(sample_root))

        entry_results = list(existing_results or [])
        if not entry_results and quantification_results_by_id is not None:
            entry_results = list(quantification_results_by_id.get(spectrum_id, []))

        return SpectrumEntry(
            live_acquisition_time=realtime_from_header if realtime_from_header is not None else 1.0,
            total_counts=total_counts,
            spectrum_id=spectrum_id,
            spectrum_relpath=pointer_rel,
            instrument_background_relpath=background_relpath,
            acquisition_details=acquisition_details,
            quantification_results=entry_results,
        )


    def _load_or_create_ledger(self) -> SampleLedger:
        """Load ledger or create/sync it from files in the spectra directory."""
        ledger_path = self._get_ledger_path()
        spectra_dir = self._get_spectra_dir()
        os.makedirs(spectra_dir, exist_ok=True)

        pointer_files = self._list_pointer_files_in_spectra_dir()
        if not pointer_files:
            # Backward compatibility: legacy datasets may only have Data.csv and no spectra/ files.
            self._populate_spectra_dir_from_data_csv()
            pointer_files = self._list_pointer_files_in_spectra_dir()

        ledger = self._load_existing_ledger()
        ledger_changed = False
        data_csv_path = os.path.join(self.sample_result_dir, cnst.DATA_FILENAME + cnst.DATA_FILEEXT)
        legacy_acq_details = load_legacy_acquisition_details_by_spectrum_id(
            data_csv_path,
            sample_result_dir=self.sample_result_dir,
            microscope_id=self.microscope_cfg.ID,
        )
        legacy_quant_results = load_legacy_quantification_results_by_spectrum_id(data_csv_path)

        if ledger is None:
            if pointer_files:
                legacy_configs = load_ledger_configs_from_legacy_json(self.sample_result_dir)
                ledger_configs = legacy_configs if legacy_configs is not None else self._build_ledger_configs()
                spectra_entries = [
                    self._build_spectrum_entry_from_pointer_file(
                        pointer_file,
                        acquisition_details_by_id=legacy_acq_details,
                        quantification_results_by_id=legacy_quant_results,
                    )
                    for pointer_file in pointer_files
                ]
                ledger = SampleLedger(
                    sample_id=self.sample_cfg.ID,
                    sample_path=os.path.abspath(self.sample_result_dir),
                    configs=ledger_configs,
                    spectra=spectra_entries,
                    quantification_configs=[
                        build_legacy_import_quantification_config(
                            sample_result_dir=self.sample_result_dir,
                            ledger_configs=ledger_configs,
                        )
                    ],
                    active_quant=0,
                )
                ledger_changed = True
            else:
                ledger = self._build_ledger_from_current_state(existing_ledger=None)
                ledger_changed = True

        existing_relpaths = {
            spectrum.spectrum_relpath
            for spectrum in ledger.spectra
            if spectrum.spectrum_relpath is not None
        }
        pointer_files = self._list_pointer_files_in_spectra_dir()
        for pointer_file in pointer_files:
            pointer_rel = str(pointer_file.resolve().relative_to(Path(self.sample_result_dir).resolve()))
            if pointer_rel in existing_relpaths:
                continue
            ledger.spectra.append(
                self._build_spectrum_entry_from_pointer_file(
                    pointer_file,
                    acquisition_details_by_id=legacy_acq_details,
                    quantification_results_by_id=legacy_quant_results,
                )
            )
            existing_relpaths.add(pointer_rel)
            ledger_changed = True

        if ledger.sample_path != os.path.abspath(self.sample_result_dir):
            ledger.sample_path = os.path.abspath(self.sample_result_dir)
            ledger_changed = True

        if self._refresh_legacy_import_payloads(
            ledger,
            data_csv_path=data_csv_path,
            legacy_quant_results=legacy_quant_results,
        ):
            ledger_changed = True

        if ledger_changed:
            ledger.to_json_file(ledger_path)

        return ledger


    def _refresh_legacy_import_payloads(
        self,
        ledger: SampleLedger,
        *,
        data_csv_path: str,
        legacy_quant_results: Dict[str, List[QuantificationResult]],
    ) -> bool:
        """Refresh legacy-import config and quant results from Data.csv when they are stale."""
        if not data_csv_path or not os.path.exists(data_csv_path):
            return False

        changed = False
        legacy_configs = load_ledger_configs_from_legacy_json(self.sample_result_dir)
        config_source = legacy_configs if legacy_configs is not None else ledger.configs
        legacy_quant_config = build_legacy_import_quantification_config(
            sample_result_dir=self.sample_result_dir,
            ledger_configs=config_source,
        )
        existing_config_idx = next(
            (i for i, config in enumerate(ledger.quantification_configs) if config.quantification_id == 0),
            None,
        )
        if existing_config_idx is None:
            ledger.quantification_configs.insert(0, legacy_quant_config)
            changed = True
        else:
            existing_config = ledger.quantification_configs[existing_config_idx]
            if existing_config.model_dump(mode="json") != legacy_quant_config.model_dump(mode="json"):
                ledger.quantification_configs[existing_config_idx] = legacy_quant_config
                changed = True

        if ledger.active_quant is None and ledger.quantification_configs:
            ledger.active_quant = 0
            changed = True

        for index, spectrum in enumerate(ledger.spectra):
            spectrum_id = str(spectrum.spectrum_id) if spectrum.spectrum_id not in (None, "") else str(index)
            replacement_results = legacy_quant_results.get(spectrum_id)
            if replacement_results is None:
                continue

            retained_results = [
                result for result in spectrum.quantification_results if result.quantification_id != 0
            ]
            merged_results = [
                result.model_copy(deep=True) for result in replacement_results
            ] + retained_results
            if [result.model_dump(mode="json") for result in spectrum.quantification_results] != [
                result.model_dump(mode="json") for result in merged_results
            ]:
                spectrum.quantification_results = merged_results
                changed = True

        return changed


    def _sync_in_memory_spectra_from_ledger(self) -> None:
        """Hydrate in-memory spectral arrays from pointer files tracked in the ledger."""
        if self.sample_result_dir is None:
            return

        ledger = self._load_or_create_ledger()
        if ledger is None or not ledger.spectra:
            return

        spectra_vals: List[np.ndarray] = []
        background_vals: List[Optional[np.ndarray]] = []
        real_times: List[float] = []
        live_times: List[float] = []
        coords: List[Dict[str, Any]] = []

        for i, spectrum in enumerate(ledger.spectra):
            if not spectrum.spectrum_relpath:
                continue

            pointer_abs = Path(ledger.sample_path, spectrum.spectrum_relpath)
            try:
                counts = SampleLedger._load_counts_from_pointer_file(pointer_abs)
            except Exception:
                continue

            spectra_vals.append(np.asarray(counts, dtype=float))

            background_vector = None
            if self.quant_cfg.use_instrument_background and spectrum.instrument_background_relpath:
                background_abs = Path(ledger.sample_path, spectrum.instrument_background_relpath)
                background_vector = self._load_background_vector_from_file(background_abs)
            background_vals.append(background_vector)

            realtime = (
                float(spectrum.live_acquisition_time)
                if spectrum.live_acquisition_time is not None
                else 1.0
            )
            real_times.append(realtime)
            live_times.append(realtime)

            acquisition_details = spectrum.acquisition_details
            x_val = ""
            y_val = ""
            x_pixel_val = ""
            y_pixel_val = ""
            par_id = ""
            frame_id = ""

            if acquisition_details is not None:
                if acquisition_details.spot_coordinates is not None:
                    machine_coords = acquisition_details.spot_coordinates.machine_coordinates
                    pixel_coords = acquisition_details.spot_coordinates.pixel_coordinates
                    if machine_coords is not None:
                        x_val = str(machine_coords.x)
                        y_val = str(machine_coords.y)
                    if pixel_coords is not None:
                        x_pixel_val = str(pixel_coords[0])
                        y_pixel_val = str(pixel_coords[1])
                par_id = str(acquisition_details.particle_id or "")
                frame_id = str(acquisition_details.frame_id or "")

            resolved_spectrum_id = str(spectrum.spectrum_id) if spectrum.spectrum_id is not None else str(i)
            coords.append(
                {
                    cnst.SP_ID_DF_KEY: resolved_spectrum_id,
                    cnst.SP_X_COORD_DF_KEY: x_val,
                    cnst.SP_Y_COORD_DF_KEY: y_val,
                    cnst.SP_X_PIXEL_COORD_DF_KEY: x_pixel_val,
                    cnst.SP_Y_PIXEL_COORD_DF_KEY: y_pixel_val,
                    cnst.PAR_ID_DF_KEY: par_id,
                    cnst.FRAME_ID_DF_KEY: frame_id,
                }
            )

        if not spectra_vals:
            return

        self.spectral_data[cnst.SPECTRUM_DF_KEY] = spectra_vals
        self.spectral_data[cnst.BACKGROUND_DF_KEY] = background_vals
        self.spectral_data[cnst.REAL_TIME_DF_KEY] = real_times
        self.spectral_data[cnst.LIVE_TIME_DF_KEY] = live_times
        self.spectral_data[cnst.COMMENTS_DF_KEY] = [None] * len(spectra_vals)
        self.spectral_data[cnst.QUANT_FLAG_DF_KEY] = [None] * len(spectra_vals)
        self.sp_coords = coords


    def _build_spectrum_entry(self, index: int, existing_results: Optional[List[QuantificationResult]] = None) -> SpectrumEntry:
        """Build one ledger spectrum entry from the current in-memory spectral data."""
        coords = self.sp_coords[index] if index < len(self.sp_coords) else {}
        spectrum_id = str(coords.get(cnst.SP_ID_DF_KEY, index))
        spectrum_vals = list(self.spectral_data[cnst.SPECTRUM_DF_KEY][index])
        real_time = self._coerce_optional_finite_float(
            self.spectral_data[cnst.REAL_TIME_DF_KEY][index] if index < len(self.spectral_data[cnst.REAL_TIME_DF_KEY]) else None
        )
        live_time = self._coerce_optional_finite_float(
            self.spectral_data[cnst.LIVE_TIME_DF_KEY][index] if index < len(self.spectral_data[cnst.LIVE_TIME_DF_KEY]) else None
        )
        live_acquisition_time = live_time if live_time is not None else real_time
        spectrum_relpath = self._resolve_or_create_spectrum_pointer(
            spectrum_id=spectrum_id,
            spectrum_vals=spectrum_vals,
            live_time=live_time,
            real_time=real_time,
        )
        background_relpath = None
        background = None
        if index < len(self.spectral_data[cnst.BACKGROUND_DF_KEY]):
            background = self.spectral_data[cnst.BACKGROUND_DF_KEY][index]
        if background is not None:
            background_relpath = self._write_manufacturer_background_vector(spectrum_id, list(background))

        raw_x = coords.get(cnst.SP_X_COORD_DF_KEY, '')
        raw_y = coords.get(cnst.SP_Y_COORD_DF_KEY, '')
        raw_x_pixel = coords.get(cnst.SP_X_PIXEL_COORD_DF_KEY, '')
        raw_y_pixel = coords.get(cnst.SP_Y_PIXEL_COORD_DF_KEY, '')

        acquisition_details = AcquisitionDetails(
            frame_id=str(coords.get(cnst.FRAME_ID_DF_KEY, '')).strip() or None,
            particle_id=self._parse_optional_int(coords.get(cnst.PAR_ID_DF_KEY, '')),
            spot_coordinates=self._build_spot_coordinates(raw_x, raw_y, raw_x_pixel, raw_y_pixel),
        )

        return SpectrumEntry(
            live_acquisition_time=live_acquisition_time if live_acquisition_time is not None else 1.0,
            total_counts=int(round(float(np.sum(spectrum_vals)))),
            spectrum_id=spectrum_id,
            spectrum_relpath=spectrum_relpath,
            instrument_background_relpath=background_relpath,
            acquisition_details=acquisition_details,
            quantification_results=list(existing_results or []),
        )


    def _build_ledger_configs(self) -> LedgerConfigs:
        """Build inline ledger configs from current analyzer configuration objects."""
        return LedgerConfigs(
            microscope_cfg=self.microscope_cfg,
            sample_cfg=self.sample_cfg,
            measurement_cfg=self.measurement_cfg,
            sample_substrate_cfg=self.sample_substrate_cfg,
            plot_cfg=self.plot_cfg,
            powder_meas_cfg=self.powder_meas_cfg,
            bulk_meas_cfg=self.bulk_meas_cfg,
            exp_stds_cfg=self.exp_stds_cfg,
        )


    def _build_ledger_from_current_state(self, existing_ledger: Optional[SampleLedger] = None) -> SampleLedger:
        """Build a ledger from current spectral data while preserving existing quantification records."""
        spectra = []
        n_spectra = len(self.spectral_data[cnst.SPECTRUM_DF_KEY])
        for index in range(n_spectra):
            existing_results = []
            if existing_ledger is not None and index < len(existing_ledger.spectra):
                existing_results = list(existing_ledger.spectra[index].quantification_results)
            spectra.append(self._build_spectrum_entry(index, existing_results=existing_results))

        existing_configs = []
        active_quant = None
        if existing_ledger is not None:
            existing_configs = list(existing_ledger.quantification_configs)
            active_quant = existing_ledger.active_quant

        return SampleLedger(
            sample_id=self.sample_cfg.ID,
            sample_path=os.path.abspath(self.sample_result_dir),
            configs=(
                existing_ledger.configs
                if existing_ledger is not None and existing_ledger.configs is not None
                else self._build_ledger_configs()
            ),
            spectra=spectra,
            quantification_configs=existing_configs,
            active_quant=active_quant,
        )


    def _ensure_quant_tracking_length(self, total_spectra: int) -> None:
        """Ensure in-memory quantification tracking lists are indexable for all spectra."""
        if len(self.spectra_quant_records) < total_spectra:
            self.spectra_quant_records.extend([None] * (total_spectra - len(self.spectra_quant_records)))
        for key in (cnst.COMMENTS_DF_KEY, cnst.QUANT_FLAG_DF_KEY):
            if len(self.spectral_data[key]) < total_spectra:
                self.spectral_data[key].extend([None] * (total_spectra - len(self.spectral_data[key])))


    def _ensure_current_quantification_run(self, force_new: bool = False) -> None:
        """Resolve the active quantification run for this launch.

        Parameters
        ----------
        force_new : bool
            If True, always create a new quantification config id.
        """
        existing_ledger = self._load_or_create_ledger()
        active_quant_config = self._get_active_quantification_config(existing_ledger)
        if active_quant_config is not None:
            self._apply_active_clustering_config(active_quant_config)

        current_sample_elements = list(self.all_els_sample)
        current_substrate_elements = list(self.all_els_substrate)
        current_options = self._build_quantification_options()
        current_reference_values = self._get_reference_values_by_el_line(active_quant_config=active_quant_config)

        candidate_id = (
            active_quant_config.quantification_id
            if active_quant_config is not None
            else self._next_quantification_id(existing_ledger)
        )
        candidate_config = self._build_quantification_config(
            quantification_id=candidate_id,
            sample_elements=current_sample_elements,
            substrate_elements=current_substrate_elements,
            options=current_options,
            reference_values_by_el_line=current_reference_values,
        )

        if (
            not force_new
            and active_quant_config is not None
            and self._quantification_configs_match(active_quant_config, candidate_config)
        ):
            self.current_quant_config = active_quant_config
            self.current_quantification_id = active_quant_config.quantification_id
            self._apply_active_clustering_config(self.current_quant_config)
            return

        if not force_new and active_quant_config is not None:
            changes = active_quant_config.fingerprint_differences(candidate_config)
            if changes:
                changed_summary = self._format_quantification_config_changes(changes)
                warnings.warn(
                    "Quantification scientific inputs changed; creating a new quantification config. "
                    f"Changed fields: {changed_summary}",
                    UserWarning,
                )

        quantification_id = self._next_quantification_id(existing_ledger)
        self.current_quant_config = self._build_quantification_config(
            quantification_id=quantification_id,
            sample_elements=current_sample_elements,
            substrate_elements=current_substrate_elements,
            options=current_options,
            reference_values_by_el_line=current_reference_values,
        )
        self.current_quantification_id = quantification_id
        self._ensure_current_clustering_run(self.current_quant_config)
        self._apply_active_clustering_config(self.current_quant_config)


    def _build_quantification_options(self) -> Dict[str, Any]:
        """Build the subset of quantification options that defines result reuse."""
        return {
            "method": self.quant_cfg.method,
            "spectrum_lims": [
                float(self.quant_cfg.spectrum_lims[0]),
                float(self.quant_cfg.spectrum_lims[1]),
            ],
            "fit_tolerance": float(self.quant_cfg.fit_tolerance),
            "use_instrument_background": bool(self.quant_cfg.use_instrument_background),
        }


    def _get_reference_values_by_el_line(
        self,
        active_quant_config: Optional[QuantificationConfig] = None,
    ) -> Dict[str, Any]:
        """Return method-dependent reference values, reusing persisted config values when possible."""
        if active_quant_config is not None and active_quant_config.reference_values_by_el_line:
            cached_reference_values = dict(sorted(active_quant_config.reference_values_by_el_line.items()))
            current_reference_values = self._extract_reference_values_from_standards(load_if_missing=False)
            if current_reference_values and current_reference_values != cached_reference_values:
                warnings.warn(
                    "Reference values in standards differ from active quantification config; "
                    "a new quantification config will be created.",
                    UserWarning,
                )
                return current_reference_values
            return cached_reference_values

        return self._extract_reference_values_from_standards(load_if_missing=True)


    def _extract_reference_values_from_standards(self, load_if_missing: bool) -> Dict[str, Any]:
        """Extract per-reference-line values from standards for the configured quantification method."""
        standards_by_line = None

        if self.quant_cfg.use_project_specific_std_dict:
            if self.XSp_std_dict is not None and not load_if_missing:
                standards_by_line = self.XSp_std_dict
            else:
                std_dict_all_modes, _ = self._load_xsp_standards()
                standards_by_line = std_dict_all_modes.get(self.measurement_cfg.mode, {})
        else:
            standards_by_line = self.XSp_std_dict
            if standards_by_line is None and self.standards_dict is not None:
                if self.measurement_cfg.mode in self.standards_dict:
                    standards_by_line = self.standards_dict[self.measurement_cfg.mode]
                else:
                    standards_by_line = self.standards_dict
            if standards_by_line is None and load_if_missing:
                std_dict_all_modes, _ = self._load_xsp_standards()
                standards_by_line = std_dict_all_modes.get(self.measurement_cfg.mode, {})

        if standards_by_line is None:
            return {}

        relevant_elements = set(self.detectable_els_sample) | set(self.detectable_els_substrate)
        reference_values_by_el_line: Dict[str, Any] = {}
        if self.quant_cfg.method == "PB":
            for el_line, std_values in standards_by_line.items():
                element = el_line.split("_", maxsplit=1)[0]
                if element not in relevant_elements:
                    continue
                mean_std = next(
                    (std for std in std_values if std.get(cnst.STD_ID_KEY) == cnst.STD_MEAN_ID_KEY),
                    None,
                )
                if mean_std is None or cnst.COR_PB_DF_KEY not in mean_std:
                    continue
                reference_values_by_el_line[el_line] = float(mean_std[cnst.COR_PB_DF_KEY])

        return dict(sorted(reference_values_by_el_line.items()))


    @staticmethod
    def _quantification_configs_match(left: QuantificationConfig, right: QuantificationConfig) -> bool:
        """Compare quantification configs by deterministic scientific-input fingerprint."""
        return left.fingerprint() == right.fingerprint()

    @staticmethod
    def _clustering_configs_match(left: LedgerClusteringConfig, right: LedgerClusteringConfig) -> bool:
        """Compare clustering configs by deterministic scientific-input fingerprint."""
        return left.fingerprint() == right.fingerprint()

    @staticmethod
    def _format_quantification_config_changes(
        changes: Dict[str, Dict[str, Any]],
        max_items: int = 8,
        max_value_length: int = 80,
    ) -> str:
        """Return a concise deterministic summary of config changes with old->new values."""
        if not changes:
            return "none"

        sorted_items = sorted(changes.items(), key=lambda item: item[0])
        formatted_entries: List[str] = []
        for path, change in sorted_items[:max_items]:
            old_val = EMXSp_Composition_Analyzer._short_repr(change.get("old"), max_value_length)
            new_val = EMXSp_Composition_Analyzer._short_repr(change.get("new"), max_value_length)
            formatted_entries.append(f"{path}: {old_val} -> {new_val}")

        remaining = len(sorted_items) - len(formatted_entries)
        if remaining > 0:
            formatted_entries.append(f"... and {remaining} more")

        return "; ".join(formatted_entries)

    @staticmethod
    def _short_repr(value: Any, max_length: int) -> str:
        """Return a stable shortened representation suitable for warning messages."""
        text = repr(value)
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."


    @staticmethod
    def _runtime_clustering_cfg_payload(clustering_cfg: LedgerClusteringConfig) -> Dict[str, Any]:
        """Convert a ledger clustering config to the runtime config payload shape."""
        payload = clustering_cfg.model_dump(mode="json")
        payload.pop("clustering_id", None)
        return payload


    def _apply_active_clustering_config(self, quant_config: QuantificationConfig) -> None:
        """Load runtime clustering options from the active clustering config of a quantification config."""
        active_clustering_config = quant_config.get_active_clustering_config()
        if active_clustering_config is None:
            raise ValueError(
                "Active quantification config does not contain an active clustering config"
            )

        self.clustering_cfg = active_clustering_config.model_copy(deep=True)
        self.ref_formulae = list(dict.fromkeys(self.clustering_cfg.ref_formulae))
        self._calc_reference_phases_df()


    def _build_clustering_config_descriptor(self, clustering_id: int) -> LedgerClusteringConfig:
        """Build a persisted clustering config descriptor for the current analysis settings."""
        return LedgerClusteringConfig(
            clustering_id=clustering_id,
            method=self.clustering_cfg.method,
            features=self.clustering_cfg.features,
            k_forced=self.clustering_cfg.k_forced,
            k_resolved=self.clustering_cfg.k_resolved,
            k_finding_method=self.clustering_cfg.k_finding_method,
            max_k=self.clustering_cfg.max_k,
            ref_formulae=list(self.clustering_cfg.ref_formulae),
            do_matrix_decomposition=self.clustering_cfg.do_matrix_decomposition,
            max_analytical_error_percent=self.clustering_cfg.max_analytical_error_percent,
            min_bckgrnd_cnts=self.clustering_cfg.min_bckgrnd_cnts,
            quant_flags_accepted=list(self.clustering_cfg.quant_flags_accepted),
        )


    def _ensure_current_clustering_run(self, quant_config: QuantificationConfig) -> None:
        """Ensure active quantification config tracks clustering run history and active config."""
        candidate_clustering_config = self._build_clustering_config_descriptor(
            clustering_id=self._next_clustering_config_id(quant_config)
        )
        active_clustering_config = quant_config.get_active_clustering_config()

        if (
            active_clustering_config is not None
            and self._clustering_configs_match(active_clustering_config, candidate_clustering_config)
        ):
            return

        if active_clustering_config is not None:
            changes = active_clustering_config.fingerprint_differences(candidate_clustering_config)
            if changes:
                changed_summary = self._format_quantification_config_changes(changes)
                warnings.warn(
                    "Clustering scientific inputs changed; appending a new clustering config to the active "
                    "quantification config. "
                    f"Changed fields: {changed_summary}",
                    UserWarning,
                )

        quant_config.clustering_configs.append(candidate_clustering_config)
        quant_config.active_clustering_cfg_index = len(quant_config.clustering_configs) - 1


    @staticmethod
    def _next_clustering_config_id(quant_config: QuantificationConfig) -> int:
        """Return the next clustering config id local to the provided quantification config."""
        if not quant_config.clustering_configs:
            return 0
        return max(cfg.clustering_id for cfg in quant_config.clustering_configs) + 1


    @staticmethod
    def _get_active_quantification_config(existing_ledger: Optional[SampleLedger]) -> Optional[QuantificationConfig]:
        """Return the active quantification config from the ledger when available."""
        if existing_ledger is None or not existing_ledger.quantification_configs:
            return None

        if existing_ledger.active_quant is not None:
            for quant_config in existing_ledger.quantification_configs:
                if quant_config.quantification_id == existing_ledger.active_quant:
                    return quant_config

        return existing_ledger.quantification_configs[-1]


    @staticmethod
    def _next_quantification_id(existing_ledger: Optional[SampleLedger]) -> int:
        """Return the next ledger-local integer quantification identifier."""
        if existing_ledger is None or not existing_ledger.quantification_configs:
            return 1
        return max(config.quantification_id for config in existing_ledger.quantification_configs) + 1


    def _build_quantification_config(
        self,
        quantification_id: int,
        sample_elements: List[str],
        substrate_elements: List[str],
        options: Dict[str, Any],
        reference_values_by_el_line: Dict[str, Any],
    ) -> QuantificationConfig:
        """Build the persisted descriptor for the current quantification run."""
        label = f"Quantification {quantification_id}"
        if isinstance(self.output_filename_suffix, str) and self.output_filename_suffix:
            label += self.output_filename_suffix
        reference_lines_by_element = QuantificationConfig.derive_reference_lines_by_element(
            reference_values_by_el_line=reference_values_by_el_line,
            preferred_lines=XSp_Quantifier.xray_quant_ref_lines,
        )
        return QuantificationConfig(
            quantification_id=quantification_id,
            label=label,
            sample_elements=sample_elements,
            substrate_elements=substrate_elements,
            options=options,
            reference_values_by_el_line=reference_values_by_el_line,
            reference_lines_by_element=reference_lines_by_element,
            clustering_configs=[],
            active_clustering_cfg_index=None,
        )


    def _build_legacy_import_quantification_config(self) -> QuantificationConfig:
        """Build the baseline quantification config stored when migrating legacy Data.csv data."""
        reference_values_by_el_line = self._extract_reference_values_from_standards(load_if_missing=True)
        reference_lines_by_element = QuantificationConfig.derive_reference_lines_by_element(
            reference_values_by_el_line=reference_values_by_el_line,
            preferred_lines=XSp_Quantifier.xray_quant_ref_lines,
        )
        return QuantificationConfig(
            quantification_id=0,
            label="Legacy import",
            sample_elements=list(self.all_els_sample),
            substrate_elements=list(self.all_els_substrate),
            options=self._build_quantification_options(),
            reference_values_by_el_line=reference_values_by_el_line,
            reference_lines_by_element=reference_lines_by_element,
            clustering_configs=[self._build_clustering_config_descriptor(clustering_id=0)],
            active_clustering_cfg_index=0,
        )


    def _upsert_current_quantification_config_on_ledger(self, ledger: SampleLedger) -> None:
        """Insert or replace the current quantification config inside the provided ledger."""
        if self.current_quant_config is None:
            return

        existing_index = next(
            (
                i
                for i, config in enumerate(ledger.quantification_configs)
                if config.quantification_id == self.current_quant_config.quantification_id
            ),
            None,
        )

        if existing_index is None:
            ledger.append_quantification_config(self.current_quant_config)
        else:
            ledger.quantification_configs[existing_index] = self.current_quant_config


    def _persist_current_quantification_config(self) -> None:
        """Persist the currently selected quantification config as the ledger active run."""
        if self.current_quant_config is None or self.current_quantification_id is None:
            return

        existing_ledger = self._load_or_create_ledger()
        ledger = self._build_ledger_from_current_state(existing_ledger)
        self._upsert_current_quantification_config_on_ledger(ledger)

        ledger.active_quant = self.current_quantification_id
        ledger.to_json_file(self._get_ledger_path())


    def _sync_existing_quantification_from_ledger(self) -> None:
        """Populate in-memory quantification slots from ledger results with the current id."""
        if self.current_quantification_id is None:
            return
        existing_ledger = self._load_or_create_ledger()
        if existing_ledger is None:
            return

        total_spectra = len(self.spectral_data[cnst.SPECTRUM_DF_KEY])
        self._ensure_quant_tracking_length(total_spectra)

        for index, spectrum in enumerate(existing_ledger.spectra[:total_spectra]):
            matching = next(
                (
                    result for result in spectrum.quantification_results
                    if result.quantification_id == self.current_quantification_id
                ),
                None,
            )
            if matching is None:
                continue

            self.spectra_quant_records[index] = matching
            self.spectral_data[cnst.COMMENTS_DF_KEY][index] = matching.comment
            self.spectral_data[cnst.QUANT_FLAG_DF_KEY][index] = matching.quant_flag


    def _persist_quantification_record(self, spectrum_index: int, quant_record: QuantificationResult, overwrite: bool = False) -> None:
        """Write one quantification result to the ledger immediately for interruption-safe progress.

        Parameters
        ----------
        overwrite : bool
            If True and a result with the same quantification_id already exists for this spectrum,
            replace it (used when requantify_only_unquantified_spectra=True).
        """
        existing_ledger = self._load_or_create_ledger()
        ledger = self._build_ledger_from_current_state(existing_ledger)

        if self.current_quant_config is None:
            raise ValueError("Current quantification config is not initialized")

        self._upsert_current_quantification_config_on_ledger(ledger)

        ledger.active_quant = self.current_quantification_id

        existing_ids = {
            existing.quantification_id
            for existing in ledger.spectra[spectrum_index].quantification_results
        }
        if quant_record.quantification_id not in existing_ids:
            ledger.append_quantification_result(spectrum_index, quant_record)
        elif overwrite:
            ledger.spectra[spectrum_index].quantification_results = [
                quant_record if r.quantification_id == quant_record.quantification_id else r
                for r in ledger.spectra[spectrum_index].quantification_results
            ]

        ledger.to_json_file(self._get_ledger_path())


    def _check_fit_quant_validity(
        self,
        is_quant_fit_valid: bool,
        bad_quant_flag: int,
        quantifier: Any,
        min_bckgrnd_ref_lines: Any
    ) -> tuple[int, str]:
        """
        Determine the quantification flag and comment for a spectrum based on fit outcomes.
    
        Parameters
        ----------
        is_quant_fit_valid : bool
            Whether the spectrum fit and quantification succeeded without errors.
        bad_quant_flag : int
            Indicator of the type of issue detected during fitting:
            - 1: poor fit
            - 2: excessively high analytical error
            - 3: excessive absorption
            - -1: non-converged fit
        quantifier : object
            The quantifier instance used for this spectrum; may be used for additional checks.
        min_bckgrnd_ref_lines : Any
            Reference value for background lines, used for further spectrum checks.
    
        Returns
        -------
        quant_flag : int
            Numerical flag representing the spectrum quality after fit/quantification.
        comment : str
            Human-readable comment describing the outcome or issue detected.
        """
        # Prefix for comments if fit was interrupted
        start_str_comments = 'Fit interrupted due to ' if not is_quant_fit_valid else ''
    
        if bad_quant_flag == 1:
            if self.verbose and is_quant_fit_valid:
                print("Flagged for poor fit")
            comment = start_str_comments + "poor fit"
            quant_flag = 4
        elif bad_quant_flag == 2:
            if self.verbose and is_quant_fit_valid:
                print("Flagged for excessively high analytical error")
            comment = start_str_comments + "excessively high analytical error"
            quant_flag = 5
        elif bad_quant_flag == 3:
            if self.verbose and is_quant_fit_valid:
                print("Flagged for excessive X-ray absorption")
            comment = start_str_comments + "excessive X-ray absorption"
            quant_flag = 6
        elif not is_quant_fit_valid:
            comment = "Fit interrupted for unknown reasons"
            quant_flag = 9
        else:
            # Fit completed with no apparent issue; check for low background counts, etc.
            _, quant_flag, comment = self._flag_spectrum_for_clustering(min_bckgrnd_ref_lines, quantifier)
    
        # If fit was good but did not converge, annotate comment safely and set flag
        if bad_quant_flag == -1 and quant_flag == 0:
            if comment:
                comment += " - Quantification did not converge."
            else:
                comment = "Quantification did not converge."
            quant_flag = -1  # Signal non-convergence
        
        return quant_flag, comment
        
    def _is_spectrum_valid_for_fitting(
        self, 
        spectrum: np.ndarray, 
        background: np.ndarray = None
    ) -> tuple[bool, int, str]:
        """
        Check if a spectrum is valid for quantification fitting.
    
        This method applies several criteria to determine if a spectrum should be processed:
          - No spectrum data present.
          - Total counts are too low.
          - Too many low-count channels in the low-energy range.
    
        For each failure, a comment and quantification flag are appended to `self.spectral_data`, and
        a message is printed if `self.verbose` is True.
    
        Parameters
        ----------
        spectrum : np.ndarray
            The spectrum data to be validated.
        background : np.ndarray, optional
            The background data (not used in this method).
    
        Returns
        -------
        is_spectrum_valid : bool
            True if the spectrum is valid for fitting, False otherwise.
        quant_flag : int
            Numerical flag representing the spectrum quality after fit/quantification.
        comment : str
            Human-readable comment describing the outcome or issue detected.
    
        Notes
        -----
        - Assumes all class attributes and keys are correctly initialized.
        - Uses constants from `cnst` for comment and flag keys.
        """
        is_spectrum_valid = True
        quant_flag = None
        comment = None
        
        if spectrum is None:
            # Check if spectrum data is present
            is_spectrum_valid = False
            comment = "No spectral data present"
            quant_flag = 1
            if self.verbose:
                print("Error during spectrum collection. No quantification was done.")
        elif np.sum(spectrum) < 0.9 * self.measurement_cfg.target_acquisition_counts:
            # Skip quantification of spectrum when counts are too low
            is_spectrum_valid = False
            comment = "Total counts too low"
            quant_flag = 2
            if self.verbose:
                print(f"Quantification skipped due to spectrum counts lower than 90% of the target counts of {self.measurement_cfg.target_acquisition_counts}")
        else:
            # Skip quantification if too many low values, which leads to errors due to imprecise fitting
            n_vals_considered = 20  # Number of data channels that must be low for spectrum to be excluded
            filter_len = 3
            en_threshold = 2  # keV
    
            # Prepare (energy, counts) pairs for the relevant region
            xy_data = zip(self.energy_vals, spectrum[self.sp_start: self.sp_end])
            # Consider only data with counts > 0 and energy < threshold
            spectrum_data_to_consider = [cnts for en, cnts in xy_data if cnts > 0 and en < en_threshold]
            # Smoothen spectrum to reduce noise
            spectrum_smooth = np.convolve(spectrum_data_to_consider, np.ones(filter_len)/filter_len, mode='same')
            # Get the n lowest values in the smoothed spectrum
            min_vals = np.sort(spectrum_smooth)[:n_vals_considered]
            min_background_threshold = self.clustering_cfg.min_bckgrnd_cnts
            if min_background_threshold is not None and all(min_vals < min_background_threshold):
                is_spectrum_valid = False
                comment = "Background counts too low"
                quant_flag = 3
                if self.verbose:
                    print(f"Quantification skipped due to at least {n_vals_considered} spectrum points with E < {en_threshold} keV having a count lower than {min_background_threshold}")
                    print("This generally indicates an excessive absorption of X-rays before they reach the detector, which compromises accurate measurements of PB ratios.")
    
        return is_spectrum_valid, quant_flag, comment
    
    
    def _flag_spectrum_for_clustering(
        self,
        min_bckgrnd_ref_lines: float,
        quantifier: Any,
    ) -> tuple[bool, int, str]:
        """
        Check spectrum validity for clustering based on substrate peak intensities and background counts.
    
        This method:
          - Flags spectra where any substrate element has a peak intensity larger than a set percentage
            of total counts.
          - Flags spectra where the minimum background counts under reference peaks are too low.
          - Appends comments and quantification flags to `self.spectral_data` using keys from `cnst`.
          - Prints warnings if `self.verbose` is True.
    
        Parameters
        ----------
        min_bckgrnd_ref_lines : float
            Minimum average counts under reference peaks in the spectrum.
        quantifier : Any
            The quantification object containing fitting information.
    
        Returns
        -------
        is_spectrum_valid : bool
            True if the spectrum passes all checks, False otherwise.
        quant_flag : int
            Numerical flag representing the spectrum quality after fit/quantification.
        comment : str
            Human-readable comment describing the outcome or issue detected.
    
        Notes
        -----
        - Assumes all class attributes and keys are correctly initialized.
        """
        is_spectrum_valid = True
    
        # Check that substrate signal is not too high
        sub_peak_int_threshold = 10  # % of total counts
        sub_peak_int_thresh_cnts = quantifier.tot_sp_counts * sub_peak_int_threshold / 100
    
        # Sum intensities from substrate peaks
        els_substrate_intensities = {el: 0 for el in self.detectable_els_substrate}  # initialise dictionary of peak intensities
        for el_line, peak_info in quantifier.fitted_peaks_info.items():
            el = el_line.split('_')[0]
            if el in self.detectable_els_substrate:
                els_substrate_intensities[el] += peak_info[cnst.PEAK_INTENSITY_KEY]
    
        # Check that no substrate element has too high intensity
        for el, peak_int in els_substrate_intensities.items():
            if peak_int > sub_peak_int_thresh_cnts:
                is_spectrum_valid = False
                comment = f"{el} {peak_int:.0f} counts > {sub_peak_int_threshold} % of total counts"
                quant_flag = 7
                if self.verbose:
                    print(f"Intensity of substrate element {el} is {peak_int:.0f} cnts, larger than {sub_peak_int_threshold}% of total counts")
                    print("This is likely to lead to large quantification errors.")
                break  # Stop if one element has too high intensity
    
        # Check that background intensity is high enough
        if is_spectrum_valid:
            comment = ""
            min_background_threshold = self.clustering_cfg.min_bckgrnd_cnts
            # Spectrum is not valid if any reference peak has average counts lower than the configured threshold.
            if min_background_threshold is not None and min_bckgrnd_ref_lines < min_background_threshold:
                is_spectrum_valid = False
                comment = (
                    f"Reference background counts too low "
                    f"({min_bckgrnd_ref_lines:.1f} < {min_background_threshold})"
                )
                quant_flag = 8
                if self.verbose:
                    print(f"Counts below a reference peak are on average < {min_background_threshold}")
                    print("This is likely to lead to large quantification errors.")
            else:
                quant_flag = 0  # Quantification is ok
    
        return is_spectrum_valid, quant_flag, comment  # Not used, but returned for completeness
    
    
    #%% Spectra acquisition and quantification routines
    # ============================================================================= 
    def _collect_spectra(
        self,
        n_spectra_to_collect: int,
        n_tot_sp_collected: int = 0,
        quantify: bool = True
    ) -> Tuple[int, bool]:
        """
        Acquire and optionally quantify spectra from particles.
    
        This method supports two operational modes:
          - Collection and quantification (default): For each spot, acquire and immediately quantify the spectrum.
          - Collection only: Only acquire spectra and update coordinates; quantification is deferred.
                              Useful when quantifying spectra separately
    
        Parameters
        ----------
        n_spectra_to_collect : int
            Number of new spectra to collect.
        n_tot_sp_collected : int, optional
            The running total of spectra already collected (default: 0).
        quantify : bool, optional
            If True, perform spectra quantification (default: True).
    
        Returns
        -------
        n_tot_sp_collected : int
            The updated total number of spectra collected after this session.
        success : bool
            False if collection was interrupted by user, or if no more particles could be found. True otherwise.
    
        Notes
        -----
        - If `quantify` is True, quantification occurs immediately after each collection.
        """
        success = False
    
        n_spectra_collected = 0
        n_spectra_init = n_tot_sp_collected

        while n_spectra_collected < n_spectra_to_collect:
            success, spots_xy_list, particle_cntr = self.EM_controller.get_XSp_coords(n_tot_sp_collected)
            
            if not success:
                break
            
            self.particle_cntr = particle_cntr
            frame_ID = self.EM_controller.current_frame_label
            
            latest_spot_id = None # For image annotations
            for i, (x, y) in enumerate(spots_xy_list):
                latest_spot_id = i
                xy_center = self.EM_controller.convert_XS_coords_to_pixels((x, y))
                value_map = {
                    cnst.SP_ID_DF_KEY: n_tot_sp_collected,
                    cnst.FRAME_ID_DF_KEY : frame_ID,
                    cnst.SP_X_COORD_DF_KEY: f'{x:.3f}',
                    cnst.SP_Y_COORD_DF_KEY: f'{y:.3f}',
                    cnst.SP_X_PIXEL_COORD_DF_KEY: f'{xy_center[0]:.2f}',
                    cnst.SP_Y_PIXEL_COORD_DF_KEY: f'{xy_center[1]:.2f}',
                }
                # Add particle ID only if not None
                if self.particle_cntr is not None:
                    value_map[cnst.PAR_ID_DF_KEY] = self.particle_cntr
                    
                self.sp_coords.append({
                    key: value_map[key]
                    for key in cnst.LIST_SPECTRUM_COORDINATES_KEYS
                    if key in value_map
                }) # Ensures any modification of keys is done at the level of LIST_SPECTRUM_COORDINATES_KEYS
                    # This allows correct loading when quantifying or analysing spectra after acquisition

                if self.verbose:
                    print_single_separator()
                    print(f'Acquiring spectrum #{n_tot_sp_collected}...')

                current_spectrum_id = str(n_tot_sp_collected)
                spectrum_relpath = self._build_spectrum_relpath(current_spectrum_id)
                manufacturer_msa_path = os.path.join(self.sample_result_dir, spectrum_relpath)

                n_tot_sp_collected += 1
                collection_time, total_counts = self._acquire_spectrum(
                    x,
                    y,
                    spectrum_id=current_spectrum_id,
                    msa_file_path=manufacturer_msa_path,
                )

                if self.verbose:
                    print(f"Acquisition took {collection_time:.2f} s")
                
                # Contamination check: skip quantification if counts are too low (only at first measurement spot)
                if i==0 and self.sample_cfg.is_particle_acquisition:
                    if total_counts < 0.95 * self.measurement_cfg.target_acquisition_counts:
                        if quantify:
                            self.spectra_quant_records.append(None)
                        if self.verbose:
                                print('Current particle is unlikely to be part of the sample.\nSkipping to the next particle.')
                                print('Increase measurement_cfg.max_acquisition_time if this behavior is undesired.')
                        break
            
            # Save image of particle, with ID of acquired XSp spots
            if latest_spot_id is not None:
                # Prepare save path
                par_cntr_str = f"_par{self.particle_cntr}" if self.particle_cntr is not None else ''
                filename = f"{self.sample_cfg.ID}{par_cntr_str}_fr{frame_ID}_xyspots"
                # Construct annotation dictionary
                im_annotations = []
                for i, xy_coords in enumerate(spots_xy_list):
                    # Skip if latest_spot_id is None or i is out of range
                    if latest_spot_id is None or i > latest_spot_id:
                        break
                
                    xy_center = self.EM_controller.convert_XS_coords_to_pixels(xy_coords)
                    if xy_center is None:
                        continue
                    
                    im_annotations.append({
                        self.EM_controller.an_text_key: (
                            str(n_tot_sp_collected - 1 - latest_spot_id + i),
                            (xy_center[0] - 30, xy_center[1] - 15)
                        ),
                        self.EM_controller.an_circle_key: (10, xy_center, -1)
                    })
                # Save image with annotations
                self.EM_controller.save_frame_image(filename, im_annotations = im_annotations)
                
            if quantify:
                self._fit_and_quantify_spectra()

            n_spectra_collected = n_tot_sp_collected - n_spectra_init
    
        return n_tot_sp_collected, success
    
    
    def _fit_and_quantify_spectra(
        self,
        quantify: bool = True,
        force_requantification: bool = False,
        requantify_only_unquantified_spectra: bool = False,
        interrupt_fits_bad_spectra: bool = True,
        num_CPU_cores: Optional[int] = None,
    ) -> None:
        """
        Fit and (optionally) quantify all collected spectra.

        Parameters
        ----------
        quantify : bool
            If False, only fits spectra (used for experimental standards). Default True.
        force_requantification : bool
            Create a new quantification run and reprocess every spectrum, even when
            an identical run already exists.
        requantify_only_unquantified_spectra : bool
            Reuse the latest matching run but reprocess only spectra that have no
            composition result (never quantified, or previously skipped/flagged).
            Overwrites the prior skipped record in the ledger. Ignored when
            force_requantification=True.
        interrupt_fits_bad_spectra : bool
            Controls early-exit behaviour during iterative spectral fitting.

            If ``True`` (default), the fit is aborted mid-iteration when poor fit quality,
            excessive analytical error, or excessive X-ray absorption is detected.
            The spectrum is stored with ``QuantificationDiagnostics.interrupted=True``
            and no composition is saved.

            If ``False``, early-exit is disabled.  Any spectrum from the active
            quantification run whose record has ``interrupted=True`` is re-quantified
            and its ledger record is overwritten with the new result.
        num_CPU_cores : Optional[int]
            Number of CPU cores for parallel fitting (non-quantify path only).
            None uses half of available cores.
        """
        
        self._sync_in_memory_spectra_from_ledger()
        tot_spectra_collected = len(self.spectral_data[cnst.SPECTRUM_DF_KEY])

        _n_cores = min(
            (num_CPU_cores if num_CPU_cores is not None else max(1, os.cpu_count() // 2)),
            os.cpu_count(),
        )

        if quantify:
            # Always bootstrap/sync ledger before quantification cycles.
            self._load_or_create_ledger()
            self._ensure_current_quantification_run(force_new=force_requantification)
            self._persist_current_quantification_config()
            self._ensure_quant_tracking_length(tot_spectra_collected)
            if force_requantification:
                indices_to_process = list(range(tot_spectra_collected))
            elif requantify_only_unquantified_spectra:
                self._sync_existing_quantification_from_ledger()
                indices_to_process = [
                    i for i in range(tot_spectra_collected)
                    if self.spectra_quant_records[i] is None
                    or self.spectra_quant_records[i].composition_atomic_fractions is None
                ]
            else:
                self._sync_existing_quantification_from_ledger()
                indices_to_process = [
                    i for i in range(tot_spectra_collected)
                    if self.spectra_quant_records[i] is None
                    or (
                        not interrupt_fits_bad_spectra
                        and getattr(self.spectra_quant_records[i].diagnostics, 'interrupted', False)
                    )
                ]
        else:
            quant_sp_cntr = len(self.spectra_quant_records)
            indices_to_process = list(range(quant_sp_cntr, tot_spectra_collected))

        n_spectra_to_quant = len(indices_to_process)
    
        if self.verbose and n_spectra_to_quant > 0:
            print_single_separator()
            quant_str = "quantification" if quantify else "fitting"
            print(f"Starting {quant_str} of {n_spectra_to_quant} spectra on up to {_n_cores} cores")
    
        # Worker returns (index, result) tuple
        def _process_one(i):
            spectrum = self.spectral_data[cnst.SPECTRUM_DF_KEY][i]
            background = (
                self.spectral_data[cnst.BACKGROUND_DF_KEY][i]
                if self.quant_cfg.use_instrument_background
                else None
            )
            sp_collection_time = self.spectral_data[cnst.LIVE_TIME_DF_KEY][i]
            sp_id = f"{i}/{tot_spectra_collected - 1}"
    
            if quantify:
                result, quant_record, quant_flag, comment = self._fit_quantify_spectrum(
                    spectrum,
                    background,
                    sp_collection_time,
                    sp_id,
                    spectrum_index=i,
                    interrupt_fits_bad_spectra=interrupt_fits_bad_spectra,
                )
            else:
                result, quant_flag, comment = self._fit_exp_std_spectrum(spectrum, background, sp_collection_time, sp_id)
                quant_record = None
    
            return i, result, quant_record, quant_flag, comment

        if quantify:
            for i in indices_to_process:
                had_prior_record = self.spectra_quant_records[i] is not None
                was_interrupted = (
                    had_prior_record
                    and getattr(self.spectra_quant_records[i].diagnostics, 'interrupted', False)
                )
                _, result, quant_record, quant_flag, comment = _process_one(i)
                self.spectra_quant_records[i] = quant_record
                self.spectral_data[cnst.COMMENTS_DF_KEY][i] = comment
                self.spectral_data[cnst.QUANT_FLAG_DF_KEY][i] = quant_flag
                if quant_record is not None:
                    overwrite = (requantify_only_unquantified_spectra and had_prior_record) or was_interrupted
                    self._persist_quantification_record(i, quant_record, overwrite=overwrite)
            return
    
        n_cores = _n_cores
    
        # Temporarily remove the analyzer to avoid pickling errors from 'loky' backend
        tmp_analyzer = None
        if hasattr(self, "EM_controller") and hasattr(self.EM_controller, "analyzer"):
            tmp_analyzer = self.EM_controller.analyzer
            del self.EM_controller.analyzer
        
        results_with_idx = []
        try:
            # Run in parallel
            results_with_idx = Parallel(n_jobs=n_cores, backend='loky')(
                delayed(_process_one)(i) for i in indices_to_process
            )
        except Exception as e:
            print(f"Parallel quantification failed ({type(e).__name__}: {e}), falling back to sequential execution.")
            # Sequential fallback, also collect results
            results_with_idx = [_process_one(i) for i in indices_to_process]
        finally:
            # Restore analyzer
            if tmp_analyzer is not None:
                self.EM_controller.analyzer = tmp_analyzer
        
        if len(results_with_idx) > 0 :
            # Sort results by original spectrum index to guarantee correct order
            results_with_idx.sort(key=lambda x: x[0])
            
            # Unpack into separate lists
            _, results_in_order, quant_records_in_order, quant_flags_in_order, comments_in_order = zip(*results_with_idx)
            
            # Convert from tuples to lists
            results_in_order = list(results_in_order)
            quant_records_in_order = list(quant_records_in_order)
            quant_flags_in_order = list(quant_flags_in_order)
            comments_in_order = list(comments_in_order)
        
            self.spectra_quant_records.extend(quant_records_in_order)
            self.spectral_data[cnst.COMMENTS_DF_KEY].extend(comments_in_order)
            self.spectral_data[cnst.QUANT_FLAG_DF_KEY].extend(quant_flags_in_order)


    #%% Find number of clusters in kmeans
    # ============================================================================= 
    def _find_optimal_k(self, compositions_df, k, compute_k_only_once = False):
        """
        Determine the optimal number of clusters for k-means.
    
        Returns
        -------
        k : int
            Optimal number of clusters.
        """
        if not k:
            # Check if there is only one single cluster, or no clusters
            is_single_cluster = EMXSp_Composition_Analyzer._is_single_cluster(compositions_df, verbose=self.verbose)
            if is_single_cluster or self.clustering_cfg.max_k <= 1:
                k = 1
            elif compute_k_only_once:
                # Get number of clusters (k) and optionally save the plot
                results_dir = self.analysis_dir if self.plot_cfg.save_plots else None
                k = EMXSp_Composition_Analyzer._get_k(
                    compositions_df, self.clustering_cfg.max_k, self.clustering_cfg.k_finding_method,
                    show_plot=self.plot_cfg.show_plots, results_dir=results_dir
                )
            else:
                # Calculate most frequent number of clusters (k) with elbow method. Does not save the plot
                k = EMXSp_Composition_Analyzer._get_most_freq_k(
                    compositions_df, self.clustering_cfg.max_k, self.clustering_cfg.k_finding_method,
                    verbose=self.verbose
                )
        elif self.verbose:
            print_single_separator()
            print(f"Number of clusters was forced to be {k}")
        return k
    
    
    @staticmethod
    def _get_most_freq_k(
        compositions_df: 'pd.DataFrame',
        max_k: int,
        k_finding_method: str,
        verbose: bool = False,
        show_plot: bool = False,
        results_dir: str = None
    ) -> int:
        """
        Determine the most frequent optimal number of clusters (k) for the given compositions.
    
        This method repeatedly runs the k-finding algorithm and selects the most robust k value.
        It loops until it finds a value of k that is at least twice as frequent as the second most frequent value,
        or until a maximum number of iterations is reached.
    
        Parameters
        ----------
        compositions_df : pd.DataFrame
            DataFrame containing the compositions to cluster.
        max_k : int
            Maximum number of clusters to test.
        k_finding_method : str
            Method used to determine the optimal k (passed to _get_k).
        verbose : bool, optional
            If True, print progress and summary information.
        show_plot : bool, optional
            If True, display plots for each k-finding run.
        results_dir : str, optional
            Directory to save plots/results (if applicable).
    
        Returns
        -------
        k : int
            The most robustly determined number of clusters.
    
        Raises
        ------
        ValueError
            If there are not enough data points to determine clusters.
    
        Notes
        -----
        - The function tries up to 5 times to find a dominant k value.
        - If a tie or ambiguity remains, it picks the smallest k with frequency ≥ half of the most frequent.
    
        """
        if len(compositions_df) < 2:
            raise ValueError("Not enough data points to determine clusters (need at least 2).")
    
        if verbose:
            print_single_separator()
            print("Computing number of clusters k...")
    
        k_found = []
        k = None
        max_iter = 5
        i = 0
    
        max_allowed_k = len(compositions_df) - 1  # KElbowVisualizer throws error if max_k > n_samples - 1
        if max_k > max_allowed_k:
            max_k = max_allowed_k
            warnings.warn(
                f"Maximum number of clusters reduced to {max_allowed_k} because number of clustered points is {max_allowed_k + 1}.",
                UserWarning
            )
    
        while k is None:
            i += 1
            for n in range(20):
                k_val = EMXSp_Composition_Analyzer._get_k(
                    compositions_df, max_k, k_finding_method,
                    show_plot=show_plot, results_dir=results_dir
                )
                if not isinstance(k_val, int) or k_val < 1:
                    continue  # skip invalid k values
                k_found.append(k_val)
    
            if not k_found:
                raise ValueError("No valid cluster counts were found.")
    
            counts = np.bincount(k_found)
            total = counts.sum()
            sorted_k = np.argsort(-counts)  # descending
            first_k = sorted_k[0]
            first_count = counts[first_k]
    
            if len(sorted_k) > 1:
                second_k = sorted_k[1]
                second_count = counts[second_k]
            else:
                second_k = None
                second_count = 0
    
            # Check if first is at least twice as common as second
            if second_count == 0 or first_count >= 2 * second_count:
                k = first_k
            elif i >= max_iter:  # max_iter reached
                # Pick smallest of all k values whose frequency is ≥ half of the most frequent
                threshold = first_count / 2
                k = min([k_val for k_val, count in enumerate(counts) if count >= threshold])
            else:
                k = None
    
        if verbose:
            print(f"Most frequent k: {first_k} (count = {first_count}, frequency = {first_count / total:.2%})")
            if second_k is not None and second_k != 0:
                print(f"Second most frequent k: {second_k} (count = {second_count}, frequency = {second_count / total:.2%})")
            if len(np.where(counts == first_count)[0]) > 1:
                print(f"Tie detected among: {np.where(counts == first_count)[0].tolist()} (choosing {k})")
    
        return int(k)
    
    
    @staticmethod
    def _get_k(
        compositions_df: 'pd.DataFrame',
        max_k: int = 6,
        method: str = 'silhouette',
        model: 'KMeans' = None,
        results_dir: str = None,
        show_plot: bool = False
    ) -> int:
        """
        Determine the optimal number of clusters for the data using visualizer methods.
    
        Parameters
        ----------
        compositions_df : pd.DataFrame
            DataFrame containing the compositions to cluster.
        max_k : int, optional
            Maximum number of clusters to test (default: 6).
        method : str, optional
            Method for evaluating the number of clusters. One of 'elbow', 'silhouette', or 'calinski_harabasz' (default: 'silhouette').
        model : KMeans or compatible, optional
            Clustering model to use (default: KMeans(n_init='auto')).
        results_dir : str, optional
            Directory to save the plot (if provided).
        show_plot : bool, optional
            If True, show the plot interactively (default: False).
    
        Returns
        -------
        optimal_k : int
            The optimal number of clusters found.
    
        Raises
        ------
        ValueError
            If an unsupported method is provided.
    
        Notes
        -----
        - Uses yellowbrick's KElbowVisualizer.
        - For 'elbow', finds the inflection point; for 'silhouette' or 'calinski_harabasz', finds the k with the highest score.
        - If cluster finding fails, returns 1 and prints a warning.
        - If `show_plot` is True, the plot is shown interactively.
        - If `results_dir` is provided, the plot is saved as 'Elbow_plot.png'.
        """
        if model is None:
            from sklearn.cluster import KMeans
            model = KMeans(n_init='auto')
    
        # Map 'elbow' to 'distortion' for yellowbrick, but keep original for logic
        user_method = method
        if method == 'elbow':
            yb_method = 'distortion'
        elif method in ['silhouette', 'calinski_harabasz']:
            yb_method = method
        else:
            raise ValueError(f"Unsupported method '{method}' for evaluating number of clusters.")
    
        plt.figure(figsize=(10, 8))
        visualizer = KElbowVisualizer(model, k=max_k, metric=yb_method, timings=True, show=False)
    
        try:
            visualizer.fit(compositions_df)
        except ValueError as er:
            warnings.warn(f"Number of clusters could not be identified due to the following error:\n{er}\nForcing k = 1.", UserWarning)
            return 1
    
        # Get optimal number of clusters
        if user_method == 'elbow':
            optimal_k = visualizer.elbow_value_
        elif user_method in ['silhouette', 'calinski_harabasz']:
            # For silhouette and calinski_harabasz, k_scores_ is indexed from k=2
            optimal_k = np.argmax(visualizer.k_scores_) + 2
            visualizer.elbow_value_ = optimal_k  # For correct plotting
    
        # Add labels
        ax1, ax2 = visualizer.axes
        ax1.set_ylabel(f'{user_method} score')
        ax1.set_xlabel('k: number of clusters')
        ax2.set_ylabel('Fit time (sec)')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
        if show_plot:
            plt.ion()
            visualizer.show()
            plt.pause(0.001)
    
        if results_dir:
            fig = visualizer.fig
            fig.savefig(os.path.join(results_dir, 'Elbow_plot.png'))
    
        if not show_plot:
            plt.close(visualizer.fig)
    
        # Set k to 1 if elbow method was unsuccessful
        if optimal_k is None:
            optimal_k = 1
    
        return int(optimal_k)
    
    
    @staticmethod
    def _is_single_cluster(
        compositions_df: 'pd.DataFrame',
        verbose: bool = False
    ) -> bool:
        """
        Determine if the data effectively forms a single cluster using k-means and silhouette analysis.
    
        This method:
          - Fits k-means with k=1 and calculates the RMS distance from the centroid.
          - Fits k-means with k=2 multiple times, keeping the best silhouette score and inertia.
          - Uses empirically determined thresholds on silhouette score, centroid distance, and inertia ratio
            to decide if the data forms a single cluster or multiple clusters.
    
        Parameters
        ----------
        compositions_df : pd.DataFrame
            DataFrame of samples (rows) and features (columns) to analyze.
        verbose : bool, optional
            If True, print detailed output of the clustering metrics.
    
        Returns
        -------
        is_single_cluster : bool
            True if the data is best described as a single cluster, False otherwise.
    
        Notes
        -----
        - Uses silhouette score and inertia ratio as main criteria.
        - Empirical thresholds: mean centroid distance < 0.025, silhouette < 0.5, or inertia ratio < 1.5.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import numpy as np
    
        if verbose:
            print_single_separator()
            print('Checking if more than 1 cluster is present...')
    
        # Fit k-means for k=1
        kmeans_1 = KMeans(n_clusters=1, random_state=0, n_init='auto')
        kmeans_1.fit(compositions_df)
        inertia_1 = kmeans_1.inertia_
        rms_distance_1 = np.sqrt(inertia_1 / len(compositions_df))
    
        # Fit k-means for k=2, keep best silhouette score and inertia
        best_silhouette_score_2 = -1
        best_inertia_2 = None
        for _ in range(10):
            kmeans_2 = KMeans(n_clusters=2, random_state=None, n_init='auto')
            labels_2 = kmeans_2.fit_predict(compositions_df)
            inertia_2 = kmeans_2.inertia_
            sil_score = silhouette_score(compositions_df, labels_2)
            if sil_score > best_silhouette_score_2:
                best_silhouette_score_2 = sil_score
                best_inertia_2 = inertia_2
    
        ratio_inertias = inertia_1 / best_inertia_2 if best_inertia_2 else float('inf')
    
        if verbose:
            print(f"RMS distance for k=1: {rms_distance_1*100:.1f}%")
            print(f"Inertia for k=1: {inertia_1:.3f}")
            print(f"Inertia for k=2: {best_inertia_2:.3f}")
            print(f"Ratio of inertia for k=1 over k=2: {ratio_inertias:.2f}")
            print(f"Silhouette Score for k=2: {best_silhouette_score_2:.2f}")
    
        # Empirical decision logic
        if rms_distance_1 < 0.03:
            is_single_cluster = True
            reason_str = 'd_rms < 3%'
        elif best_silhouette_score_2 < 0.5:
            is_single_cluster = True
            reason_str = 's < 0.5'
        elif best_silhouette_score_2 > 0.6:
            is_single_cluster = False
            reason_str = 's > 0.6'
        elif ratio_inertias < 1.5:
            is_single_cluster = True
            reason_str = 'ratio of inertias < 1.5'
        else:
            is_single_cluster = False
            reason_str = 'ratio of inertias > 1.5 and 0.5 < s < 0.6'
    
        if verbose:
            if is_single_cluster:
                print(reason_str + ": The data effectively forms a single cluster.")
            else:
                print(reason_str + ": The data forms multiple clusters.") 
    
        return is_single_cluster
    #%% Clustering operations
    # ============================================================================= 
    def _run_kmeans_clustering(self, k, compositions_df):
        """
        Run k-means clustering multiple times and select the best solution by silhouette score.
    
        Returns
        -------
        kmeans : KMeans
            The best fitted KMeans instance.
        labels : np.ndarray
            Cluster labels for each composition.
        sil_score : float
            Best silhouette score obtained.
        """
        if k > 1:
            n_clustering_eval = 20  # Number of clustering evaluations to run
            best_sil_score = -np.inf  # Initialise best silhouette score
            for _ in range(n_clustering_eval):
                # K-means is not ideal for clusters with varying sizes/densities. Consider alternatives (e.g., GMM).
                is_clustering_ok = False
                max_loops_nonneg_silh = 50  # Max loops to find clustering solutions with no negative silhouette values
                n_loop = 0
                while not is_clustering_ok and n_loop < max_loops_nonneg_silh:
                    n_loop += 1
                    kmeans, labels = self._get_clustering_kmeans(k, compositions_df)
                    silhouette_vals = silhouette_samples(compositions_df, labels)
                    if np.all(silhouette_vals > 0):
                        # Clustering is accepted only if all silhouette values are positive (no wrong clustering)
                        is_clustering_ok = True
                    sil_score = silhouette_score(compositions_df, labels)
                if sil_score > best_sil_score:
                    best_kmeans = kmeans
                    best_labels = labels
                    best_sil_score = sil_score
            return best_kmeans, best_labels, best_sil_score
        else:
            # Clustering with k = 1 is trivial, and has no silhouette score
            kmeans, labels = self._get_clustering_kmeans(k, compositions_df)
            return kmeans, labels, np.nan
    

    def _prepare_composition_dataframes(self, compositions_list_at, compositions_list_w):
        """
        Convert lists of compositions to DataFrames for clustering.
    
        Returns
        -------
        compositions_df : pd.DataFrame
            DataFrame of compositions for clustering (feature set selected).
        compositions_df_other_fr : pd.DataFrame
            DataFrame of compositions in the alternate fraction representation.
        """
        # Substitute nan with 0
        if self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
            compositions_df = pd.DataFrame(compositions_list_at).fillna(0)
            compositions_df_other_fr = (pd.DataFrame(compositions_list_w)).fillna(0) 
        elif self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
            compositions_df = pd.DataFrame(compositions_list_w).fillna(0)
            compositions_df_other_fr = (pd.DataFrame(compositions_list_at)).fillna(0)
            
        return compositions_df, compositions_df_other_fr
    
    
    def _get_clustering_kmeans(
        self,
        k: int,
        compositions_df: 'pd.DataFrame'
    ) -> Tuple['KMeans', 'np.ndarray']:
        """
        Perform k-means clustering on the given compositions.
    
        Parameters
        ----------
        k : int
            The number of clusters to find.
        compositions_df : pd.DataFrame
            DataFrame of samples (rows) and features (columns) to cluster.
    
        Returns
        -------
        kmeans : KMeans
            The fitted KMeans object.
        labels : np.ndarray
            Array of cluster (phase) labels for each composition point.
    
        Raises
        ------
        ValueError
            If clustering is unsuccessful due to invalid data or parameters.
    
        Notes
        -----
        - Uses k-means++ initialization and scikit-learn's default settings.
        - n_init='auto' requires scikit-learn >= 1.2.0.
        """
        try:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto')
            # Perform clustering. Returns labels (array of cluster (= phase) ID each composition point belongs to)
            labels = kmeans.fit_predict(compositions_df)
        except Exception as e:
            raise ValueError(f"Clustering unsuccessful due to the following error:\n{e}")
    
        return kmeans, labels

    
    def _get_clustering_dbscan(
        self,
        compositions_df: 'pd.DataFrame'
    ) -> Tuple['np.ndarray', int]:
        """
        Perform DBSCAN clustering on the given compositions.
        CURRENTLY NOT SUPPORTED
    
        Parameters
        ----------
        compositions_df : pd.DataFrame
            DataFrame of samples (rows) and features (columns) to cluster.
    
        Returns
        -------
        labels : np.ndarray
            Array of cluster labels for each composition point. Noise points are labeled as -1.
        num_labels : int
            Number of unique clusters found (excluding noise points).
    
        Raises
        ------
        ValueError
            If clustering is unsuccessful due to invalid data or parameters.
    
        Notes
        -----
        - Uses eps=0.1 and min_samples=1 as DBSCAN parameters by default.
        - The number of clusters excludes noise points (label -1).
        """
        try:
            dbscan = DBSCAN(eps=0.1, min_samples=1)
            labels = dbscan.fit_predict(compositions_df)
        except Exception as e:
            raise ValueError(f"Clustering unsuccessful due to the following error:\n{e}")
    
        # Get the number of unique labels, excluding noise (-1)
        num_labels = len(set(labels)) - (1 if -1 in labels else 0)
    
        return labels, num_labels


    def _persist_resolved_k_on_active_clustering_config(self, resolved_k: Optional[int]) -> None:
        """Persist resolved k for the active clustering config of the active quantification run."""
        if resolved_k is None:
            return

        resolved_k_int = int(resolved_k)
        self.clustering_cfg.k_resolved = resolved_k_int

        if self.current_quant_config is not None:
            active_clustering_config = self.current_quant_config.get_active_clustering_config()
            if active_clustering_config is not None and active_clustering_config.k_resolved != resolved_k_int:
                active_clustering_config.k_resolved = resolved_k_int
                self._persist_current_quantification_config()
            return

        ledger = self._load_or_create_ledger()
        active_quant_config = self._get_active_quantification_config(ledger)
        if active_quant_config is None:
            return

        active_clustering_config = active_quant_config.get_active_clustering_config()
        if active_clustering_config is None or active_clustering_config.k_resolved == resolved_k_int:
            return

        active_clustering_config.k_resolved = resolved_k_int
        ledger.to_json_file(self._get_ledger_path())


    def _compute_cluster_statistics(self, compositions_df, compositions_df_other_fr, centroids, labels):
        """
        Compute statistics for each cluster, including WCSS, standard deviations, and centroids
        in terms of both atomic and mass fractions.
    
        Returns
        -------
        wcss_per_cluster : list
            Within-Cluster Sum of Squares for each cluster.
        rms_dist_cluster : list
            Standard deviation of distances to centroid for each cluster.
        rms_dist_cluster_other_fr : list
            Standard deviation of distances in alternate fraction representation.
        n_points_per_cluster : list
            Number of points in each cluster.
        els_std_dev_per_cluster : list
            Elemental standard deviations within each cluster.
        els_std_dev_per_cluster_other_fr : list
            Elemental standard deviations in alternate fraction representation.
        centroids_other_fr : list
            Centroids in alternate fraction representation.
        max_cl_rmsdist : float
            Maximum standard deviation across all clusters.
        """
        wcss_per_cluster = []
        rms_dist_cluster = []
        rms_dist_cluster_other_fr = []
        n_points_per_cluster = []
        els_std_dev_per_cluster = []
        els_std_dev_per_cluster_other_fr = []
        centroids_other_fr = []
        for i, centroid in enumerate(centroids):
            # Save data using the elemental fraction employed as feature
            cluster_points = compositions_df[labels == i].to_numpy()
            n_points_per_cluster.append(len(cluster_points))
            if len(cluster_points) > 1:
                els_std_dev_per_cluster.append(np.std(cluster_points, axis=0, ddof=1))
            else:
                # Append NaN or zero or skip
                els_std_dev_per_cluster.append(np.full(cluster_points.shape[1], np.nan))
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            wcss_per_cluster.append(np.sum(distances ** 2))
            rms_dist_cluster.append(np.sqrt(np.mean(distances ** 2)))
    
            # Save data also for the other elemental fraction type
            cluster_points_other = compositions_df_other_fr[labels == i].to_numpy()
            centroid_other_fr = np.mean(cluster_points_other, axis=0)
            centroids_other_fr.append(centroid_other_fr)
            if len(cluster_points) > 1:
                els_std_dev_per_cluster_other_fr.append(np.std(cluster_points_other, axis=0, ddof=1))
            else:
                # Append NaN or zero or skip
                els_std_dev_per_cluster_other_fr.append(np.full(cluster_points_other.shape[1], np.nan))
            distances_other_fr = np.linalg.norm(cluster_points_other - centroid_other_fr, axis=1)
            rms_dist_cluster_other_fr.append(np.sqrt(np.mean(distances_other_fr ** 2)))
            
        max_cl_rmsdist = max(rms_dist_cluster)
        
        return (wcss_per_cluster, rms_dist_cluster, rms_dist_cluster_other_fr, n_points_per_cluster,
                els_std_dev_per_cluster, els_std_dev_per_cluster_other_fr, centroids_other_fr, max_cl_rmsdist)
    
    
    #%% Data compositional analysis
    # =============================================================================     
    def analyse_data(self, max_analytical_error_percent, k=None, compute_k_only_once=False):
        """
        Analyse quantified spectra, perform clustering, assign candidate phases and mixtures, and save results.
    
        This function orchestrates the workflow:
          1. Selects good compositions for clustering.
          2. Prepares DataFrames for clustering.
          3. Determines the optimal number of clusters (k).
          4. Runs clustering and computes cluster statistics.
          5. Assigns candidate phases and detects mixtures.
          6. Saves results and related plots.
    
        Parameters
        ----------
        max_analytical_error : float or None
            Maximum allowed analytical error for a composition to be considered valid, expressed as w%.
        k : int, optional
            Number of clusters to use (if not provided, determined automatically).
        compute_k_only_once : bool, optional
            If True, compute k only once; otherwise, use the most frequent k.
    
        Returns
        -------
        success : bool
            True if analysis was successful, False otherwise.
        max_cl_rmsdist : float
            Maximum standard deviation across clusters.
        min_conf : float or None
            Minimum confidence among assigned candidate phases.
        """
        # Ensure in-memory state is fully synced from the ledger before analysis.
        # This is essential when analyse_data is called without a preceding
        # run_quantification (i.e. analysis-only path): we need both the spectral
        # arrays and the typed QuantificationResult records populated in memory.
        if self.sample_result_dir is not None:
            self._sync_in_memory_spectra_from_ledger()
            existing_ledger = self._load_or_create_ledger()
            # Resolve current_quantification_id from the ledger when not already set
            # (analysis-only callers skip run_quantification so it is never assigned).
            if self.current_quantification_id is None and existing_ledger is not None:
                if existing_ledger.active_quant is not None:
                    self.current_quantification_id = existing_ledger.active_quant
            tot = len(self.spectral_data[cnst.SPECTRUM_DF_KEY])
            if tot > 0:
                self._ensure_quant_tracking_length(tot)
                self._sync_existing_quantification_from_ledger()

        # 1. Select compositions to use for clustering
        if max_analytical_error_percent is not None:
            max_analytical_error = max_analytical_error_percent / 100
        else:
            max_analytical_error = max_analytical_error_percent
        (compositions_list_at, compositions_list_w, unused_compositions_list,
         df_indices, n_datapts) = self._select_good_compositions(max_analytical_error)
        n_datapts_used = len(compositions_list_at)
    
        if n_datapts_used < 5:
            print_single_separator()
            print(f"Only {n_datapts_used} spectra were considered 'good', but a minimum of 5 data points are required for clustering.")
            # Print additional messages with how many spectra were discarded for which reason
            self._report_n_discarded_spectra(n_datapts, max_analytical_error)
            return False, 0, 0  # zeroes are placeholders
    
        if self.verbose:
            print_single_separator()
            print('Spectra selection:')
            print(f"{n_datapts_used} data points are used, out of {n_datapts} collected spectra.")
            self._report_n_discarded_spectra(n_datapts, max_analytical_error)
    
        # 2. Make analysis directory to save results
        self._make_analysis_dir()
    
        self._save_analysis_config_summary()

        # 3. Prepare DataFrames for clustering
        compositions_df, compositions_df_other_fr = self._prepare_composition_dataframes(compositions_list_at, compositions_list_w)

        # 4. Perform clustering
        if k is None:
            if self.clustering_cfg.k_finding_method == "forced":
                k = self.clustering_cfg.k_forced
                if k is None:
                    raise ValueError("k_finding_method='forced' requires k_forced to be set")
            else:
                # Recompute k for non-forced methods on each analysis run.
                k = None
        if self.clustering_cfg.method == 'kmeans':
            k = self._find_optimal_k(compositions_df, k, compute_k_only_once)
            self._persist_resolved_k_on_active_clustering_config(k)
            kmeans, labels, sil_score = self._run_kmeans_clustering(k, compositions_df)
            centroids = kmeans.cluster_centers_
            wcss = kmeans.inertia_
        elif self.clustering_cfg.method == 'dbscan':
            # labels, num_labels = self._get_clustering_dbscan(compositions_df)
            print('Clustering via DBSCAN is not implemented yet')
            return False, 0, 0  # zeroes are placeholders
    
        # 5. Compute cluster statistics
        (wcss_per_cluster, rms_dist_cluster, rms_dist_cluster_other_fr, 
         n_points_per_cluster, els_std_dev_per_cluster, els_std_dev_per_cluster_other_fr,
         centroids_other_fr, max_cl_rmsdist) = self._compute_cluster_statistics(
            compositions_df, compositions_df_other_fr, centroids, labels
        )
    
        # 6. Assign candidate phases
        min_conf, max_raw_confs, refs_assigned_df = self._assign_reference_phases(centroids, rms_dist_cluster)
    
        # 7. Assign mixtures
        if self.clustering_cfg.do_matrix_decomposition:
            clusters_assigned_mixtures = self._assign_mixtures(
                k, labels, compositions_df, rms_dist_cluster, max_raw_confs, n_points_per_cluster
            )
        else:
            clusters_assigned_mixtures = []
    
        # 8. Save and store results
        if self.is_acquisition:
            # When collecting, save collected spectra, their quantification, and to which cluster they are assigned
            self._save_collected_data(labels, df_indices, backup_previous_data=True, include_spectral_data=True)
        else:
            # During analysis of Data.csv, save the compositions, together with their assigned phases, in the per-config analysis directory
            self._save_collected_data(labels, df_indices, backup_previous_data=True, include_spectral_data=False)
    
        self._save_result_and_stats(
            centroids, els_std_dev_per_cluster, centroids_other_fr, els_std_dev_per_cluster_other_fr,
            n_points_per_cluster, wcss_per_cluster, rms_dist_cluster, rms_dist_cluster_other_fr,
            refs_assigned_df, wcss, sil_score, n_datapts, max_analytical_error, clusters_assigned_mixtures
        )
    
        # 9. Save plots
        if self.plot_cfg.save_plots:
            self._save_plots(kmeans, compositions_df, centroids, labels, els_std_dev_per_cluster, unused_compositions_list)
    
        return True, max_cl_rmsdist, min_conf
    
    
    def _select_good_compositions(self, max_analytical_error):
        """
        Select compositions for clustering, filtering out those with high analytical error or bad quantification flags.
    
        Returns
        -------
        compositions_list_at : list
            List of atomic fractions for good spectra.
        compositions_list_w : list
            List of mass fractions for good spectra.
        unused_compositions_list : list
            List of compositions not used for clustering (for plotting).
        df_indices : list
            Indices of rows used for phase identification.
        n_datapts : int
            Total number of spectra considered.
        """
        # Initialise counters for spectra filtered out
        self.n_sp_too_low_counts = 0
        self.n_sp_too_high_an_err = 0
        self.n_sp_bad_quant = 0
    
        compositions_list_at = []
        compositions_list_w = []
        unused_compositions_list = []
        df_indices = []
        n_datapts = len(self.spectra_quant_records)
    
        for i in range(n_datapts):
            record = self.spectra_quant_records[i]
            if record is not None and record.composition_atomic_fractions is not None:
                is_comp_ok = True
                spectrum_quant_result_at = dict(record.composition_atomic_fractions)
                spectrum_quant_result_w = dict(record.composition_weight_fractions)
                analytical_error = float(record.analytical_error)
                quant_flag = self.spectral_data[cnst.QUANT_FLAG_DF_KEY][i]

                # Check if composition was flagged as bad during quantification
                if quant_flag not in self.clustering_cfg.quant_flags_accepted:
                    is_comp_ok = False
                    self.n_sp_bad_quant += 1
    
                elif max_analytical_error is None:
                    # Analytical error check is disabled
                    is_comp_ok = True
                    pass
    
                # Check if analytical error is too high
                elif analytical_error < - (max_analytical_error + self.undetectable_an_er) or analytical_error > max_analytical_error:
                    is_comp_ok = False
                    self.n_sp_too_high_an_err += 1
    
                # Append composition to list of used or unused datapoints
                if is_comp_ok:
                    df_indices.append(i)
                    # Construct dictionary that includes all elements that are supposed to be in the sample.
                    comp_at = {el: spectrum_quant_result_at.get(el, 0) for el in self.all_els_sample}
                    compositions_list_at.append(comp_at)
                    comp_w = {el: spectrum_quant_result_w.get(el, 0) for el in self.all_els_sample}
                    compositions_list_w.append(comp_w)
                else:
                    # Collect unused data points to show them in the clustering plot
                    if self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
                        comp = [spectrum_quant_result_at.get(el, 0) for el in self.all_els_sample]
                    elif self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
                        comp = [spectrum_quant_result_w.get(el, 0) for el in self.all_els_sample]
                    unused_compositions_list.append(comp)
            else:
                self.n_sp_too_low_counts += 1
    
        return compositions_list_at, compositions_list_w, unused_compositions_list, df_indices, n_datapts


    def _correlate_centroids_to_refs(
        self,
        centroids: 'np.ndarray',
        cluster_radii: 'np.ndarray',
        ref_phases_df: 'pd.DataFrame'
    ) -> Tuple[List[float], 'pd.DataFrame']:
        """
        Correlate each cluster centroid to candidate phases and compute confidence scores.
    
        For each centroid, selects all candidate phases within a hypersphere of radius
        max(0.1, 5 * cluster_radius) around the centroid, and computes a confidence score
        for each reference. The highest confidence for each cluster is stored.
    
        Parameters
        ----------
        centroids : np.ndarray, shape (n_clusters, n_features)
            Array of cluster centroids (in elemental fraction space).
        cluster_radii : np.ndarray, shape (n_clusters,)
            Array of standard deviations (radii) for each cluster.
        ref_phases_df : pd.DataFrame
            DataFrame where each row is a candidate phase (elemental fractions).
    
        Returns
        -------
        max_raw_confs : list of float
            List of maximum confidence scores for each cluster.
        refs_assigned_df : pd.DataFrame
            DataFrame with reference names and their confidences for each cluster.
    
        Notes
        -----
        - Only candidate phases within 5 times the cluster radius (or at least 0.1) from the centroid are considered.
        - Confidence is computed using EMXSp_Composition_Analyzer._get_ref_confidences.
        """
        # Get all candidate phase compositions as a numpy array
        all_ref_phases = ref_phases_df.to_numpy()
        refs_dict = []        # For DataFrame of references and their confidences
        max_raw_confs = []    # For convergence checks
    
        # For each cluster, assign to reference(s) if present and calculate confidence
        for centroid, radius in zip(centroids, cluster_radii):
            # Calculate distances from centroid to each candidate phase
            distances = np.linalg.norm(all_ref_phases - centroid, axis=1)
            # Select all candidate phases within 5*radius (min 0.1) of centroid
            indices = np.where(distances < max(0.1, 5 * radius))[0]
            # Get chemical formulae and compositions of selected candidate phases
            ref_names = [self.ref_formulae[i] for i in indices]
            ref_phases = [all_ref_phases[i] for i in indices]
            # Calculate confidences based on distance between centroid and reference
            max_raw_conf, refs_dict_row = EMXSp_Composition_Analyzer._get_ref_confidences(
                centroid, ref_phases, ref_names
            )
            # Store maximum confidence for this cluster
            max_raw_confs.append(max_raw_conf)
            # Store dictionary of reference names and confidences for this cluster
            refs_dict.append(refs_dict_row)
    
        # Create DataFrame with information on candidate phases assigned to clusters
        refs_assigned_df = pd.DataFrame(refs_dict)
    
        return max_raw_confs, refs_assigned_df


    def _assign_reference_phases(self, centroids, rms_dist_cluster):
        """
        Assign candidate phases to clusters if reference formulae are provided.
    
        Returns
        -------
        min_conf : float or None
            Minimum confidence among all clusters assigned to a reference.
        max_raw_confs : list or None
            Maximum raw confidence scores for each cluster.
        refs_assigned_df : pd.DataFrame or None
            DataFrame of reference assignments.
        """
        min_conf = None
        max_raw_confs = None
        refs_assigned_df = None
        if self.ref_formulae is not None:
            # Correlate calculated centroids to the candidate phases
            max_raw_confs, refs_assigned_df = self._correlate_centroids_to_refs(
                centroids, rms_dist_cluster, self.ref_phases_df
            )
            # Get lowest value among the highest confidences assigned to each cluster, used for convergence
            if len(max_raw_confs) > 0:
                max_confs_num = [conf for conf in max_raw_confs if conf is not None]
                if len(max_confs_num) > 0:
                    min_conf = min(max_confs_num)
        return min_conf, max_raw_confs, refs_assigned_df


    @staticmethod
    def _get_ref_confidences(
        centroid: 'np.ndarray',
        ref_phases: 'np.ndarray',
        ref_names: List[str]
    ) -> Tuple[Optional[float], Dict]:
        """
        Compute confidence scores for candidate phases near a cluster centroid.
    
        For each candidate phase within a cluster's neighborhood, this function:
          - Computes the Euclidean distance to the centroid.
          - Assigns a confidence score using a Gaussian function of the distance.
          - Reduces confidences if multiple references are nearby, to account for ambiguity.
          - Returns a dictionary of references and their confidences (sorted by confidence), and the maximum raw confidence.
    
        Parameters
        ----------
        centroid : np.ndarray
            Cluster centroid in feature space (shape: n_features,).
        ref_phases : np.ndarray
            Array of candidate phase compositions (shape: n_refs, n_features).
        ref_names : list of str
            Names of candidate phases.
    
        Returns
        -------
        max_raw_conf : float or None
            Maximum confidence score among the references, or None if no reference is close.
        refs_dict : Dict
            Dictionary of reference names and their confidences, sorted by confidence.
            Keys: 'Cnd1', 'CS_cnd1', 'Cnd2', 'CS_cnd2', etc.
    
        Notes
        -----
        - Only confidences above 1% are included in the output dictionary.
        - The confidence spread (sigma) is set to 0.03 for the Gaussian.
        - Nearby references reduce each other's confidence using a secondary Gaussian weighting.
        """
        if ref_phases == [] or len(ref_phases) == 0:
            # No candidate phase is close enough to the centroid
            refs_dict = {f'{cnst.CND_DF_KEY}1': np.nan, f'{cnst.CS_RAW_CND_DF_KEY}1': np.nan, '{cnst.CS_CND_DF_KEY}1': np.nan}
            max_raw_conf = None
        else:
            # Calculate distances from centroid to each candidate phase
            distances = np.linalg.norm(ref_phases - centroid, axis=1)
    
            # Assign confidence using a Gaussian function (sigma = 0.03)
            raw_confidences = np.exp(-distances**2 / (2 * 0.03**2))
    
            # Reduce confidences for ambiguity if multiple references are close
            weights_conf = np.exp(-(1 - raw_confidences)**2 / (2 * 0.3**2))
            weights_conf /= np.sum(weights_conf)  # Normalize
    
            # Adjust confidences by their weights
            confidences = raw_confidences * weights_conf
    
            # Get maximum raw confidence
            max_raw_conf = float(np.max(raw_confidences))
    
            # Sort references by confidence, descending
            sorted_indices = np.argsort(-confidences)
            sorted_ref_names = np.array(ref_names)[sorted_indices]
            sorted_confidences = confidences[sorted_indices]
            sorted_raw_confs = raw_confidences[sorted_indices]
    
            # Build dictionary of references and confidences (only those > 1%)
            refs_dict = {}
            for i, (ref_name, conf, conf_raw) in enumerate(zip(sorted_ref_names, sorted_confidences, sorted_raw_confs)):
                if conf_raw > 0.05:
                    refs_dict[f'{cnst.CND_DF_KEY}{i+1}'] = ref_name
                    refs_dict[f'{cnst.CS_CND_DF_KEY}{i+1}'] = np.round(conf, 2)
                    refs_dict[f'{cnst.CS_RAW_CND_DF_KEY}{i+1}'] = np.round(conf_raw, 2)
    
        return max_raw_conf, refs_dict


    #%% Binary cluster decomposition
    # =============================================================================       
    def _assign_mixtures(self, k, labels, compositions_df, rms_dist_cluster, max_raw_confs, n_points_per_cluster):
        """
        Determine if clusters are mixtures or single phases, using candidate phases and NMF if needed.
    
        Returns
        -------
        clusters_assigned_mixtures : list
            List of mixture assignments for each cluster.
            
        Potential improvements
        ----------------------
        Instead of using the cluster standard deviation, use covariance of elemental fractions
        to discern clusters that may originate from binary phase mixtures or solid solutions.
        """
        clusters_assigned_mixtures = []
        for i in range(k):
            # Get compositions of data points included in cluster as np.array (only detectable elements)
            cluster_data = compositions_df[self.detectable_els_sample].iloc[labels == i].values
            max_mix_conf = 0
            mixtures_dicts = []
            
            # # Use log-ratio transformations, which map the data from the simplex to real Euclidean space
            # if len(cluster_data) > 1:
            #     # Suppose X is your n × m array of normalized compositions
            #     X_clr = clr(cluster_data + 1e-6) # to avoid zeroes
            #     # Compute covariance matrix on CLR-transformed data
            #     cov_clr = np.cov(X_clr.T, rowvar=True)
            #     print(cov_clr)
                
            #     # 5. Compute correlation matrix
            #     std_dev = np.sqrt(np.diag(cov_clr))
            #     corr_matrix = cov_clr / np.outer(std_dev, std_dev)
                
            #     # 6. Plot covariance heatmap
            #     plt.figure(figsize=(10, 4))
                
            #     plt.subplot(1, 2, 1)
            #     sns.heatmap(cov_clr, xticklabels=self.detectable_els_sample, yticklabels=self.detectable_els_sample,
            #                 cmap='coolwarm', center=0, annot=True, fmt=".3f")
            #     plt.title('Covariance matrix (CLR space)')
                
            #     # 7. Plot correlation heatmap
            #     plt.subplot(1, 2, 2)
            #     sns.heatmap(corr_matrix, xticklabels=self.detectable_els_sample, yticklabels=self.detectable_els_sample,
            #                 cmap='coolwarm', center=0, annot=True, fmt=".3f")
            #     plt.title('Correlation matrix (CLR space)')
                
            #     plt.tight_layout()
            #     plt.show()
            
            max_rmsdist_single_cluster = 0.03
            if rms_dist_cluster[i] < max_rmsdist_single_cluster:
                if max_raw_confs is None or len(max_raw_confs) < 1:
                    is_cluster_single_phase = n_points_per_cluster[i] > 3
                elif max_raw_confs[i] is not None and max_raw_confs[i] > 0.5:
                    is_cluster_single_phase = True
                else:
                    is_cluster_single_phase = False
            else:
                is_cluster_single_phase = False
    
            if is_cluster_single_phase:
                # Cluster determined to stem from a single phase
                pass
            elif len(self.ref_formulae) > 1:
                max_mix_raw_conf, mixtures_dicts = self._identify_mixture_from_refs(cluster_data, cluster_ID = i)
                max_mix_conf = max(max_mix_conf, max_mix_raw_conf)
            if not is_cluster_single_phase and max_mix_conf < 0.5:
                mix_nmf_conf, mixture_dict = self._identify_mixture_nmf(cluster_data, cluster_ID = i)
                if mixture_dict is not None:
                    mixtures_dicts.append(mixture_dict)
                max_mix_conf = max(max_mix_conf, mix_nmf_conf)
            clusters_assigned_mixtures.append(mixtures_dicts)
        return clusters_assigned_mixtures
    
    
    def _identify_mixture_from_refs(self, X: 'np.ndarray', cluster_ID: int = None) -> Tuple[float, List[Dict]]:
        """
        Identify mixtures within a cluster by testing all pairs of candidate phases using constrained optimization.
    
        For each possible pair of candidate phases, tests if the cluster compositions (X)
        can be well described by a linear combination of the two candidate phases, using
        non-negative matrix factorization (NMF) with fixed bases.
    
        Parameters
        ----------
        X : np.ndarray
            Cluster data (compositions), shape (n_samples, n_features).
        cluster_ID : int
            Current cluster ID. Used for violin plot name
    
        Returns
        -------
        max_confidence : float
            The highest confidence score among all tested mixtures.
        mixtures_dicts : list of Dict
            List of mixture descriptions for all successful reference pairs.
        cluster_ID : int
            Current cluster ID. Used for violin plot name
    
        Notes
        -----
        - Each mixture is described by a dictionary, as returned by _get_mixture_dict_with_conf.
        - Only pairs with acceptable reconstruction error are included.
        - The confidence metric and acceptance criteria are defined in _get_mixture_dict_with_conf.
        """
        # Generate all possible pairs of candidate phases
        ref_pair_combinations = list(itertools.combinations(range(len(self.ref_phases_df)), 2))
    
        mixtures_dicts = []
        max_confidence = 0
    
        for ref_comb in ref_pair_combinations:
            # Get the names of the candidate phases in this pair
            ref_names = [self.ref_formulae[ref_i] for ref_i in ref_comb]
    
            # Ratio of weights of references, for molar concentrations of parent phases
            ref_w_r = self.ref_weights_in_mixture[ref_comb[0]] / self.ref_weights_in_mixture[ref_comb[1]]
    
            # Get matrix of basis vectors (H) for the two candidate phases
            H = np.array([
                self.ref_phases_df[self.detectable_els_sample].iloc[ref_i].values
                for ref_i in ref_comb
            ])
            
            # Perform NMF with fixed H to fit the cluster data as a mixture of the two candidate phases
            W, _ = self._nmf_with_constraints(X, n_components=2, fixed_H=H)
    
            # Compute reconstruction error for the fit
            recon_er = self._calc_reconstruction_error(X, W, H)
    
            # If the pair yields an acceptable reconstruction error, store the result
            pair_dict, conf = self._get_mixture_dict_with_conf(W, ref_w_r, recon_er, ref_names, cluster_ID)
            if pair_dict is not None:
                mixtures_dicts.append(pair_dict)
                max_confidence = max(max_confidence, conf)
    
        return max_confidence, mixtures_dicts


    def _calc_reconstruction_error(
        self,
        X: 'np.ndarray',
        W: 'np.ndarray',
        H: 'np.ndarray'
    ) -> float:
        """
        Calculate the reconstruction error for a matrix factorization X ≈ W @ H.
    
        The error metric is an exponential penalty (with parameter alpha) applied to the
        absolute difference between X and its reconstruction W @ H, normalized by the
        number of elements in X. This penalizes large deviations more strongly.
    
        Parameters
        ----------
        X : np.ndarray
            Original data matrix of shape (m, n).
        W : np.ndarray
            Weight matrix of shape (m, k).
        H : np.ndarray
            Basis matrix of shape (k, n).
    
        Returns
        -------
        normalized_norm : float
            The normalized exponential reconstruction error.
    
        Notes
        -----
        - The penalty parameter alpha is set to 15 by default.
        """
        # Compute the approximation WH
        WH = np.dot(W, H)
    
        # Compute the Frobenius norm of the difference (X - WH), using an exponential form to penalize deviations more strongly
        alpha = 15
        norm = np.sum(np.exp(alpha * np.abs(X - WH)) - 1)
    
        # Get dimensions of the matrix X
        m, n = X.shape
    
        # Normalize the error by the number of entries
        normalized_norm = norm / (m * n)
    
        return normalized_norm
    
    
    def _get_mixture_dict_with_conf(
        self,
        W: 'np.ndarray',
        ref_w_r: float,
        reconstruction_error: float,
        ref_names: List[str],
        cluster_ID: int = None
    ) -> Tuple[Optional[Dict], float]:
        """
        Evaluate if a cluster is a mixture of two candidate phases, and compute a confidence score.
    
        If the reconstruction error is below a set threshold, computes a confidence score and
        transforms the NMF coefficients into molar fractions. Returns a dictionary describing
        the mixture and the confidence score.
    
        Parameters
        ----------
        W : np.ndarray
            Matrix of NMF coefficients for each point in the cluster (shape: n_points, 2).
        ref_w_r : float
            Ratio of weights of the two candidate phases (for molar concentration conversion).
        reconstruction_error : float
            Reconstruction error for the mixture fit.
        ref_names : list of str
            Names of the two candidate phases.
        cluster_ID : int
            Current cluster ID. Used for violin plot name
    
        Returns
        -------
        mixture_dict : Dict or None
            Dictionary with mixture information if fit is acceptable, else None.
            Keys: 'refs', 'conf_score', 'mean', 'stddev'.
        conf : float
            Confidence score for this mixture (0 if not acceptable).
    
        Notes
        -----
        - The minimum acceptable reconstruction error is set empirically to 2 (a.u.).
        - Confidence is calculated as a Gaussian function of the reconstruction error (sigma=0.5).
        - Molar fractions are derived from NMF coefficients and normalized.
        - Only the first component's mean and stddev are returned in the dictionary.
        """
        # Set a minimum reconstruction error threshold for accepting mixtures
        min_acceptable_recon_error = 2  # Empirically determined
        
        save_violin_plot = getattr(
            self.powder_meas_cfg,
            "is_known_powder_mixture_meas",
            False,
        )
        
        if reconstruction_error < min_acceptable_recon_error or save_violin_plot:
            # Calculate confidence score: 0.66 when error is 0.5 (empirical)
            gauss_sigma = 0.5
            conf = np.exp(-reconstruction_error**2 / (2 * gauss_sigma**2))
    
            # Transform NMF coefficients into molar fractions (see documentation for derivation)
            W_mol_frs = []
            for c1, c2 in W:
                # x1, x2 are the molar fractions
                x2 = c2 * ref_w_r / (1 - c2 * (1 - ref_w_r))
                x1 = c1 * (1 + x2 * (1 / ref_w_r - 1))
                W_mol_frs.append([x1, x2])
            W_mol_frs = np.array(W_mol_frs)
    
            # Calculate mean and standard deviation of molar fractions (not normalized)
            mol_frs_norm_means = np.mean(W_mol_frs, axis=0)
            mol_frs_norm_stddevs = np.std(W_mol_frs, axis=0)
            
            if save_violin_plot:
                self._save_violin_plot_powder_mixture(W_mol_frs, ref_names, cluster_ID)
            
            # Store mixture information
            mixture_dict = {
                cnst.REF_NAME_KEY: ref_names,
                cnst.CONF_SCORE_KEY: conf,
                cnst.MOLAR_FR_MEAN_KEY: mol_frs_norm_means[0],
                cnst.MOLAR_FR_STDEV_KEY: mol_frs_norm_stddevs[0]
            }
        else:
            mixture_dict = None
            conf = 0
    
        return mixture_dict, conf
    
    
    def _nmf_with_constraints(
        self,
        X: 'np.ndarray',
        n_components: int,
        fixed_H: 'np.ndarray' = None
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Perform Non-negative Matrix Factorization (NMF) with optional constraints on the factor matrices.
    
        This function alternates between optimizing two non-negative matrices W and H, such that X ≈ W @ H:
          - If H is fixed (provided via fixed_H), only W is updated.
          - If H is not fixed, both W and H are updated via alternating minimization.
    
        Constraints:
          - Both W and H are non-negative.
          - The rows of both W (sum of coefficients) and H (sum of elemental fractions) sum to 1.
          - Sparsity regularization (L1) is applied to H when it is updated, to favor bases with limited elements.
    
        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            The input matrix to be factorized, where m is the number of samples and n is the number of elements.
        n_components : int
            The number of latent components (rank of the decomposition). For binary mixtures, n_components=2.
        fixed_H : np.ndarray or None, optional
            If provided, a fixed basis matrix of shape (n_components, n). If None, H is updated during optimization.
    
        Returns
        -------
        W : np.ndarray, shape (m, n_components)
            The non-negative coefficient matrix learned during the factorization.
        H : np.ndarray, shape (n_components, n)
            The non-negative basis matrix learned during the factorization.
            If fixed_H is provided, this matrix is not modified.
    
        Notes
        -----
        - Uses alternating minimization, solving for one matrix while keeping the other fixed.
        - Convergence is based on the Frobenius norm of the change in W and H between iterations.
        - Stops when the change is smaller than the specified tolerance (convergence_tol = 1e-3), or max_iter is reached.
        - Regularization may be applied to H (if it is updated) to encourage sparsity and avoid all elements being present in both parent phases.
        """
        max_iter = 1000
        convergence_tol = 1e-3  # Algorithm converges when change in coefficients or el_fr is less than 0.1%
        lambda_H = 0  # Regularization parameter for sparsity in H. Set >0 to favor sparse basis matrix. Found to work better when not applied.
    
        # Initialize W and H with non-negative random values if not provided
        W = np.random.rand(X.shape[0], n_components)
        if fixed_H is None:
            H = np.random.rand(n_components, X.shape[1])  # Initialize H if not provided
        else:
            H = fixed_H
    
        prev_W, prev_H = np.inf, np.inf
        convergence = np.inf
        i = 0
    
        while convergence > convergence_tol and i < max_iter:
            # Solve for W with H fixed (or fixed_H provided)
            W_var = cp.Variable((X.shape[0], n_components), nonneg=True)
            objective_W = cp.Minimize(cp.sum_squares(X - W_var @ H))
            constraints_W = [cp.sum(W_var, axis=1) == 1]
            problem_W = cp.Problem(objective_W, constraints_W)
            problem_W.solve(solver=cp.ECOS)
            W = W_var.value # Update W
            
            # If H is not fixed, solve for H as well (alternating minimization)
            if fixed_H is None:
                H_var = cp.Variable((n_components, X.shape[1]), nonneg=True)
                objective_H = cp.Minimize(
                    cp.sum_squares(X - W @ H_var) + lambda_H * cp.norm1(H_var)
                )
                constraints_H = [cp.sum(H_var, axis=1) == 1]
                problem_H = cp.Problem(objective_H, constraints_H)
                problem_H.solve(solver=cp.ECOS)
                H = H_var.value # Update H
    
            # Compute convergence based on the changes in W and H
            convergence_W = np.linalg.norm(W - prev_W, 'fro')
            convergence_H = np.linalg.norm(H - prev_H, 'fro') if fixed_H is None else 0
            convergence = max(convergence_W, convergence_H)
    
            prev_W, prev_H = W, H
            i += 1
    
        return W, H
        
    
    def _identify_mixture_nmf(
        self,
        X: 'np.ndarray',
        n_components: int = 2,
        cluster_ID: int = None
    ) -> Tuple[float, Optional[Dict]]:
        """
        Identify a mixture within a cluster using unconstrained NMF (Non-negative Matrix Factorization).
    
        This method fits the cluster data X to n_components using NMF with constraints (rows of W and H sum to 1),
        evaluates the reconstruction error, and if acceptable, returns a dictionary describing the mixture and a confidence score.
    
        Parameters
        ----------
        X : np.ndarray
            Cluster data (compositions), shape (n_samples, n_features).
        n_components : int, optional
            Number of components (phases) to fit (default: 2).
    
        Returns
        -------
        conf : float
            Confidence score for the mixture (0 if not acceptable).
        mixture_dict : Dict or None
            Dictionary describing the mixture if reconstruction is acceptable, else None.
        cluster_ID : int
            Current cluster ID. Used for violin plot name
    
        Notes
        -----
        - The confidence and mixture dictionary are computed using _get_mixture_dict_with_conf.
        - Elemental fractions lower than 0.5% are set to 0 using _get_pretty_formulas_nmf.
        
        Potential improvements
        ----------------------
        Consider re-running algorithms with fixed zeroes after purifying compositions
        """
        mixture_dict = None
        conf = 0
    
        # Run NMF, constraining coefficients and values of bases to add up to 1
        W, H = self._nmf_with_constraints(X, n_components)
    
        # Compute the reconstruction error
        recon_er = self._calc_reconstruction_error(X, W, H)
    
        # Get human-readable formulas and weights for the NMF bases
        ref_names, ref_weights = self._get_pretty_formulas_nmf(H, n_components)
    
        # Calculate ratio of reference weights, needed to compute molar fractions of parent phases
        ref_w_r = ref_weights[0] / ref_weights[1]

        # If pair of bases yields an acceptable reconstruction error, store the mixture info
        mixture_dict, conf = self._get_mixture_dict_with_conf(W, ref_w_r, recon_er, ref_names, cluster_ID)  # Returns (None, 0) if error is too high

        return conf, mixture_dict
    
    
    def _get_pretty_formulas_nmf(
        self,
        phases: 'np.ndarray',
        n_components: int
    ) -> Tuple[List[str], List[float]]:
        """
        Generate human-readable (pretty) formulas from NMF bases, accounting for data noise.
    
        For each component, filters out small fractions, constructs a composition dictionary,
        and returns a formula string and a weight or atom count, depending on the clustering feature.
    
        Parameters
        ----------
        phases : np.ndarray
            Array of shape (n_components, n_elements), each row is a basis vector from NMF.
        n_components : int
            Number of components (phases) to process.
    
        Returns
        -------
        ref_names : list of str
            List of pretty formula strings for each phase.
        ref_weights : list of float
            List of weights (for mass fractions) or atom counts (for atomic fractions), for each phase.
    
        Notes
        -----
        - Fractions below 0.5% are set to zero for formula construction.
        - Uses `Composition` to generate formulas and weights.
        - For mass fractions, the weight of the phase is used.
        - For atomic fractions, the total atom count in the formula is used.
        """
        ref_names = []
        ref_weights = []
    
        for i in range(n_components):
            # Filter out too small fractions (set <0.5% to zero)
            frs = phases[i, :].copy()
            frs[frs < 0.005] = 0
    
            # Build dictionary for the parent phase composition
            fr_dict = {el: fr for el, fr in zip(self.detectable_els_sample, frs)}
    
            # Generate a Composition object based on the selected feature type
            if self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
                comp = Composition().from_weight_dict(fr_dict)
            elif self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
                comp = Composition(fr_dict)
    
            # Get integer formula and construct a pretty formula
            formula = comp.get_integer_formula_and_factor()[0]
            ref_integer_comp = Composition(formula)
            min_at_n = min(ref_integer_comp.get_el_amt_dict().values())
            pretty_at_frs = {el: round(n / min_at_n, 1) for el, n in ref_integer_comp.get_el_amt_dict().items()}
            pretty_comp = Composition(pretty_at_frs)
            pretty_formula = pretty_comp.formula
            ref_names.append(pretty_formula)
    
            # Store weight or atom count for the phase
            if self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
                ref_weights.append(pretty_comp.weight)
            elif self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
                n_atoms_in_formula = sum(pretty_comp.get_el_amt_dict().values())
                ref_weights.append(n_atoms_in_formula)
    
        return ref_names, ref_weights
    

    def _build_mixtures_df(
        self,
        clusters_assigned_mixtures: List[List[Dict]]
    ) -> 'pd.DataFrame':
        """
        Build a DataFrame summarizing mixture assignments for each cluster.
    
        For each cluster, sorts mixture dictionaries by confidence score and extracts:
          - candidate phase names (as a comma-separated string)
          - Confidence score
          - Molar ratio (mean / (1 - mean))
          - Mean and standard deviation of the main component's molar fraction
    
        Parameters
        ----------
        clusters_assigned_mixtures : list of list of Dict
            Outer list: clusters; inner list: mixture dictionaries for each cluster.
    
        Returns
        -------
        mixtures_df : pd.DataFrame
            DataFrame summarizing mixture assignments for each cluster.
            Columns: Mix1, CS_mix1, Mol_Ratio1, Icomp_Mol_Fr_Mean1, Stddev1, etc.
    
        Notes
        -----
        - If no mixtures are assigned for a cluster, the entry will be an empty dictionary.
        - The DataFrame is intended for addition to Clusters.csv or similar summary files.
        """
        mixtures_strings_dict = []
        for mixtures_dict in clusters_assigned_mixtures:
            if mixtures_dict:
                # Sort mixture dictionaries by decreasing confidence score
                sorted_mixtures = sorted(mixtures_dict, key=lambda x: -x[cnst.CONF_SCORE_KEY])
                cluster_mix_dict = {}
                for i, mixture_dict in enumerate(sorted_mixtures, start=1):
                    cluster_mix_dict[f'{cnst.MIX_DF_KEY}{i}'] = ', '.join(mixture_dict[cnst.REF_NAME_KEY])
                    cluster_mix_dict[f'{cnst.CS_MIX_DF_KEY}{i}'] = np.round(mixture_dict[cnst.CONF_SCORE_KEY], 2)
                    cluster_mix_dict[f'{cnst.MIX_MOLAR_RATIO_DF_KEY}{i}'] = np.round(mixture_dict[cnst.MOLAR_FR_MEAN_KEY] / (1 - mixture_dict[cnst.MOLAR_FR_MEAN_KEY]), 2)
                    cluster_mix_dict[f'{cnst.MIX_FIRST_COMP_MEAN_DF_KEY}{i}'] = np.round(mixture_dict[cnst.MOLAR_FR_MEAN_KEY], 2)
                    cluster_mix_dict[f'{cnst.MIX_FIRST_COMP_STDEV_DF_KEY}{i}'] = np.round(mixture_dict[cnst.MOLAR_FR_STDEV_KEY], 2)
                mixtures_strings_dict.append(cluster_mix_dict)
            else:
                mixtures_strings_dict.append({})
    
        # Create DataFrame to add to Clusters.csv
        mixtures_df = pd.DataFrame(mixtures_strings_dict)
    
        return mixtures_df
    
    
    #%% Run algorithms
    # =============================================================================     
    def run_collection_and_quantification(
        self,
        quantify: bool = True,
    ) -> Tuple[bool, bool]:
        """
        Perform iterative collection (and optional quantification) of spectra, followed by phase analysis and convergence check.
    
        This method:
          - Iteratively collects spectra in batches (and quantifies them if `quantify` is True).
          - After each batch, saves collected data and (if quantification is enabled) performs phase analysis (clustering).
          - Checks for convergence based on clustering statistics and confidence.
          - Stops early if convergence is achieved and minimum spectra is reached, or if no more particles are available.
    
        Parameters
        ----------
        quantify : bool, optional
            If True (default), spectra are quantified after each batch and clustering is performed.
            If False, only spectra collection is performed; quantification and clustering are skipped.
    
        Returns
        -------
        is_analysis_successful : bool
            if quantify == True: True if analysis was successful, False otherwise.
            if quantify == False: True if collection of target number of spectra was successful, False otherwise.
        is_converged : bool
            True if phase identification converged to acceptable errors, False otherwise.
    
        Notes
        -----
        - During experimental standard collection, "quantify" in fact determines whether spectra are "fitted" in-situ
        - Saves data after each batch to prevent data loss.
        - Prints a summary and processing times at the end.
        """
        tot_n_spectra = 0  # Total number of collected spectra
        max_n_sp_per_iter = 10  # Max spectra to collect per iteration (for saving in between)
        tot_spectra_to_collect = self.max_n_spectra
        n_spectra_to_collect = min(max_n_sp_per_iter, tot_spectra_to_collect, self.min_n_spectra)
        is_converged = False
        is_analysis_successful = False
        is_acquisition_successful = True
        is_exp_std_measurement = self.exp_stds_cfg.is_exp_std_measurement
        is_spectral_quant = quantify and not is_exp_std_measurement
        if is_spectral_quant:
            self._initialise_std_dict() # Initialise dictionary of standards to (optionally) pass onto XSp_Quantifier. Only used with known powder mixtures
        
        if quantify:
            if is_exp_std_measurement:
                quant_str = ' and fitting'
            else:
                quant_str = ' and quantification'
        else:
            quant_str = ''

        
        if self.verbose:
            print_double_separator()
            print(f"Starting collection{quant_str} of {tot_spectra_to_collect} spectra.")
        
        while tot_n_spectra < tot_spectra_to_collect:
            if self.verbose:
                print_double_separator()
                print(f"Collecting{quant_str} {n_spectra_to_collect} spectra...")
    
            # Collect the next batch of spectra (and quantify if requested)
            tot_n_spectra, is_acquisition_successful = self._collect_spectra(
                n_spectra_to_collect,
                n_tot_sp_collected=tot_n_spectra,
                quantify=is_spectral_quant
            )
    
            # Save temporary data file to avoid data loss
            if self.is_acquisition:
                self._save_collected_data(None, None, backup_previous_data=False, include_spectral_data=True)
                
            if self.verbose:
                print_single_separator()
                print(f"{tot_n_spectra}/{tot_spectra_to_collect} spectra collected and saved.")
                
            # Collect additional spectra in next iteration
            n_spectra_to_collect = min(
                max_n_sp_per_iter,
                tot_spectra_to_collect - tot_n_spectra,
                self.min_n_spectra
            )
            
            if quantify and tot_n_spectra > 0:
                if is_exp_std_measurement:
                    # Fit spectra and check if target number of good spectra has been collected
                    is_analysis_successful, is_converged = self._evaluate_exp_std_fit(tot_n_spectra)
                else:
                    # Perform clustering analysis and check for convergence
                    is_analysis_successful, is_converged = self._evaluate_clustering_convergence(tot_n_spectra, n_spectra_to_collect)
                    
                if is_converged:
                    break
                
                
            # Stop if no more particles are available on the sample
            if not is_acquisition_successful:
                if self.verbose:
                    print("Acquisition interrupted.")
                    if self.sample_cfg.is_particle_acquisition:
                        print(f'Not enough particles were found on the sample to collect all {tot_spectra_to_collect} spectra.')
                    elif self.sample_cfg.is_grid_acquisition:
                        print(f'The specified spectrum spacing did not allow to collect all {tot_spectra_to_collect} spectra.\n'
                              "Change spacing in bulk_meas_cfg to collect more spectra.")
                break
    
        print_double_separator()
        print('Sample ID: %s' % self.sample_cfg.ID)
        par_str = f' over {self.particle_cntr} particles' if self.sample_cfg.is_particle_acquisition else ''
        print(f'{tot_n_spectra} spectra were collected{par_str}.')
        process_time = (time.time() - self.start_process_time) / 60
        print(f'Total compositional analysis time: {process_time:.1f} min')
        print_single_separator()
    
        if is_spectral_quant:
            if is_analysis_successful:
                if is_converged:
                    print('Clustering converged to small errors. All phases identified with confidence higher than 0.8.')
                else:
                    print('Phases could not be identified with confidence higher than 0.8.')
    
                self.print_results()
    
            elif not is_acquisition_successful:
                print('This did not allow to determine which phases are present in the sample.')
            else:
                print(f'Phases could not be identified with the allowed maximum of {self.max_n_spectra} collected spectra.')
                is_analysis_successful = False
                is_converged = False
        else:
            is_analysis_successful = is_acquisition_successful
    
        return is_analysis_successful, is_converged
    
    
    def _evaluate_clustering_convergence(
        self,
        tot_n_spectra: int,
        n_spectra_to_collect: int
    ) -> Tuple[bool, bool]:
        """
        Evaluate whether compositional clustering analysis has converged.
    
        This method checks the results of the clustering analysis after a given number of spectra 
        have been collected. It determines whether the analysis has converged based on the 
        clustering standard deviation and minimum confidence, and whether additional spectra 
        should be collected.
    
        Parameters
        ----------
        tot_n_spectra : int
            Total number of spectra collected so far.
        n_spectra_to_collect : int
            Total number of spectra to be collected.
    
        Returns
        -------
        Tuple[bool, bool]
            A tuple containing:
            - is_analysis_successful (bool): Whether the clustering analysis ran successfully.
            - is_converged (bool): Whether the compositional analysis has converged.
    
        Raises
        ------
        RuntimeError
            If `analyse_data` returns unexpected results.
        """
    
        if self.verbose:
            print_double_separator()
            print(f"Analysing phases after collection of {tot_n_spectra} spectra...")
    
        try:
            is_analysis_successful, max_cl_rmsdist, min_conf = self.analyse_data(
                self.clustering_cfg.max_analytical_error_percent,
                k=self.clustering_cfg.k_forced if self.clustering_cfg.k_finding_method == "forced" else None
            )
        except Exception as e:
            raise RuntimeError(f"Error during clustering analysis: {e}") from e
    
        # Default value in case convergence check is skipped
        is_converged = False
    
        if is_analysis_successful:
            if self.verbose:
                print_double_separator()
                print("Clustering analysis performed")
    
            # Check whether phase identification converged
            try:
                is_converged = self._is_comp_analysis_converged(max_cl_rmsdist, min_conf)
            except Exception as e:
                raise RuntimeError(f"Error while checking convergence: {e}") from e
    
            if tot_n_spectra >= self.min_n_spectra:
                if is_converged:
                    return is_analysis_successful, is_converged
                elif self.verbose and n_spectra_to_collect > 0:
                    print("Compositional analysis did not converge, more spectra will be collected.")
            elif tot_n_spectra >= self.max_n_spectra:
                print(f"Maximum allowed number of {self.max_n_spectra} was acquired.")
            else:
                if self.verbose:
                    print(f"Collecting additional spectra to reach minimum number of {self.min_n_spectra}.")
    
        elif self.verbose:
            print("Clustering analysis unsuccessful.")
            if n_spectra_to_collect > 0:
                print(", more spectra will be collected.")
    
        return is_analysis_successful, is_converged
    
    
    def _is_comp_analysis_converged(
        self,
        rms_dist: float,
        min_conf: Optional[float]
    ) -> bool:
        """
        Determine if the clustering analysis has converged based on cluster statistics.
        Used when collecting and quantifying spectra in real time.
    
        Convergence criteria:
          - If no candidate phases are present or assigned (min_conf is None), require cluster RMS point-to-centroid distance to be  < 2.5%.
          - If candidate phases are assigned, require minimum confidence > 0.8 and cluster standard deviation < 3%.
    
        Parameters
        ----------
        rms_dist : float
            Maximum RMS point-to-centroid distance among clusters (fractional units, e.g., 0.025 for 2.5%).
        min_conf : float or None
            Minimum confidence among all clusters assigned to candidate phases. If None, no references are assigned.
    
        Returns
        -------
        is_converged : bool
            True if convergence criteria are met, False otherwise.
    
        Notes
        -----
        - The thresholds are empirically determined for robust phase identification.
        """
        if min_conf is None:
            # No candidate phases present or assigned; require tighter cluster homogeneity
            is_converged = rms_dist < 0.025
        else:
            # Require high confidence and allow slightly larger within-cluster spread
            is_converged = (min_conf > 0.8) and (rms_dist < 0.03)
    
        return is_converged
    

    def run_quantification(
        self,
        force_requantification: bool = False,
        requantify_only_unquantified_spectra: bool = False,
        interrupt_fits_bad_spectra: bool = True,
        num_CPU_cores: Optional[int] = None,
    ) -> None:
        """
        Perform quantification of all collected spectra and save the results.
    
        Parameters
        ----------
        force_requantification : bool, optional
            If True, quantifies all spectra again even when the same quantification
            settings were already used before (creates a new run).
        requantify_only_unquantified_spectra : bool, optional
            Reuse the latest matching run but reprocess only spectra with no composition
            result (never quantified or previously skipped/flagged). Overwrites prior
            skipped records. Ignored when force_requantification=True.
        interrupt_fits_bad_spectra : bool, optional
            Controls early-exit behaviour during iterative spectral fitting.

            If ``True`` (default), the fit is aborted mid-iteration when poor fit quality,
            excessive analytical error, or excessive X-ray absorption is detected.
            The spectrum is stored with ``QuantificationDiagnostics.interrupted=True``
            and no composition is saved.

            If ``False``, early-exit is disabled.  Any spectrum from the active
            quantification run whose record has ``interrupted=True`` is re-quantified
            and its ledger record is overwritten with the new result.
        num_CPU_cores : Optional[int], optional
            Number of CPU cores for parallel fitting (non-quantify path only).
            None uses half of available cores.
        """
        self._initialise_std_dict()
        self._fit_and_quantify_spectra(
            force_requantification=force_requantification,
            requantify_only_unquantified_spectra=requantify_only_unquantified_spectra,
            interrupt_fits_bad_spectra=interrupt_fits_bad_spectra,
            num_CPU_cores=num_CPU_cores,
        )
        self._save_collected_data(None, None, backup_previous_data=True, include_spectral_data=True)
        
    
    def run_exp_std_collection(
        self,
        fit_during_collection: bool,
        update_std_library: bool
    ) -> None:
        """
        Collect, fit, and optionally update the library of experimental standards.
    
        This method automates the acquisition and fitting of spectra from experimental 
        standards, ensuring that all required elemental fractions are defined before 
        proceeding.
    
        Parameters
        ----------
        fit_during_collection : bool
            If True, spectra will be fitted in real-time during collection.
            If False, fitting must be performed after collection.
        update_std_library : bool
            If True, the experimental standard library will be updated with the 
            newly fitted PB ratios.
    
        Raises
        ------
        ValueError
            If `self.exp_stds_cfg.is_exp_std_measurement` is not set to True.
        KeyError
            If any element in `self.sample_cfg.elements` is missing from 
            `self.exp_stds_cfg.w_frs`.
        """
    
        if not self.exp_stds_cfg.is_exp_std_measurement:
            raise ValueError(
                "Experimental standard collection mode is not active. "
                "Set `self.exp_stds_cfg.is_exp_std_measurement = True` before running."
            )
        
        # Ensure all elemental fractions are defined in the experimental standard configuration
        missing = [el for el in self.sample_cfg.elements if el not in self.exp_stds_cfg.w_frs]
        if missing:
            raise KeyError(
                f"The following elements are missing from `exp_stds_cfg.formula`: "
                f"{', '.join(str(m) for m in missing)}. "
                f"Ensure the formula contains all elements defined in `self.sample_cfg.elements`."
            )
        
        if self.verbose:
            print_double_separator()
            print(f"Experimental standard acquisition of {self.sample_cfg.ID}")
        
        # Run collection and quantification (fitting optionally performed during collection)
        self._th_peak_energies = {} # Initialise
        self.run_collection_and_quantification(quantify=fit_during_collection)
        
        # Fit standards and save results
        std_ref_lines, results_df, Z_sample = self._fit_stds_and_save_results(backup_previous_data=False)
        
        # Optionally update the standards library with the new results
        if update_std_library and std_ref_lines is not None and len(std_ref_lines) > 0: 
            self._update_standard_library(std_ref_lines, results_df, Z_sample)
        
    #%% Save Plots
    def _save_plots(
        self,
        kmeans: 'KMeans',
        compositions_df: 'pd.DataFrame',
        centroids: 'np.ndarray',
        labels: 'np.ndarray',
        els_std_dev_per_cluster: list,
        unused_compositions_list: list
    ) -> None:
        """
        Generate and save clustering and silhouette plots for the clustering analysis.
    
        This method:
          - Saves a silhouette plot if more than one cluster is present.
          - Determines which elements to include in the clustering plot (max 3 for 3D).
          - Excludes elements as specified in plot configuration.
          - Warns if only one element is available for plotting.
          - Saves a clustering plot (2D or 3D) using either a custom or default plotting function.
    
        Parameters
        ----------
        kmeans : KMeans
            Fitted KMeans clustering object.
        compositions_df : pd.DataFrame
            DataFrame of sample compositions used for clustering.
        centroids : np.ndarray
            Array of cluster centroids (shape: n_clusters, n_features).
        labels : np.ndarray
            Cluster labels for each sample.
        els_std_dev_per_cluster : list
            List of standard deviations for each element in each cluster.
        unused_compositions_list : list
            List of compositions excluded from clustering.
    
        Returns
        -------
        None
    
        Notes
        -----
        - Only up to 3 elements can be plotted; others are excluded.
        - If only one element is left after exclusions, no plot is generated.
        - Uses either a custom or default plotting function as configured.
        """
        # Silhouette plot (only if more than one cluster)
        if len(centroids) > 1:
            EMXSp_Composition_Analyzer._save_silhouette_plot(
                kmeans, compositions_df, self.analysis_dir, show_plot=self.plot_cfg.show_plots
            )
    
        # Determine which elements can be used for plotting
        can_plot_clustering = True
    
        # Elements for plotting (excluding those set for exclusion)
        els_for_plot = list(set(self.detectable_els_sample) - set(self.plot_cfg.els_excluded_clust_plot))
        els_excluded_clust_plot = list(set(self.all_els_sample) - set(els_for_plot))
        n_els = len(els_for_plot)
    
        if n_els == 1:
            # Cannot plot with only 1 detectable element
            can_plot_clustering = False
            print_single_separator()
            warnings.warn("Cannot generate clustering plot with a single element.", UserWarning)
            if len(self.detectable_els_sample) > 1:
                print('Too many elements were excluded from the clustering plot via the use of "els_excluded_clust_plot".')
                print(f'Consider removing one or more among the list: {self.plot_cfg.els_excluded_clust_plot}')
        elif n_els > 3:
            # Only 3 elements can be plotted at once (for 3D)
            els_excluded_clust_plot += els_for_plot[3:]
            els_for_plot = els_for_plot[:3]
    
        # Determine indices to remove for excluded elements
        indices_to_remove = [self.all_els_sample.index(el) for el in els_excluded_clust_plot]
        # Update values to exclude the selected elements
        els_for_plot = [el for i, el in enumerate(self.all_els_sample) if i not in indices_to_remove]
        centroids = np.array([
            [coord for i, coord in enumerate(row) if i not in indices_to_remove]
            for row in centroids
        ])
        els_std_dev_per_cluster = [
            [stddev for i, stddev in enumerate(row) if i not in indices_to_remove]
            for row in els_std_dev_per_cluster
        ]
        unused_compositions_list = [
            [fr for i, fr in enumerate(row) if i not in indices_to_remove]
            for row in unused_compositions_list
        ]
    
        # Generate and save the clustering plot if possible
        if can_plot_clustering:
            # List of lists, where each list is populated with the atomic fractions of one element in all data points
            els_comps_list = compositions_df[els_for_plot].to_numpy().T
    
            # Use custom or default plotting function
            if self.plot_cfg.use_custom_plots:
                custom_plotting._save_clustering_plot_custom_3D(
                    els_for_plot, els_comps_list, centroids, labels,
                    els_std_dev_per_cluster, unused_compositions_list,
                    self.clustering_cfg.features, self.ref_phases_df,
                    self.ref_formulae, self.plot_cfg.show_plots, self.sample_cfg.ID
                )
            else:
                self._save_clustering_plot(
                    els_for_plot, els_comps_list, centroids, labels,
                    els_std_dev_per_cluster, unused_compositions_list
                )
        elif self.verbose:
            print('Clusters were not plotted because only one detectable element was present in the sample.')
            print(f"Elements {calibs.undetectable_els} cannot be detected at the employed instrument.")
            
            
    def _save_clustering_plot(
        self,
        elements: List[str],
        els_comps_list: 'np.ndarray',
        centroids: 'np.ndarray',
        labels: 'np.ndarray',
        els_std_dev_per_cluster: list,
        unused_compositions_list: list
    ) -> None:
        """
        Generate and save a 2D or 3D clustering plot with centroids, standard deviation ellipses/ellipsoids,
        unused compositions, and candidate phases.
    
        Parameters
        ----------
        elements : list of str
            List of element symbols to plot (max 3).
        els_comps_list : np.ndarray
            Array of shape (n_elements, n_samples) with elemental fractions for each sample (used for clustering).
        centroids : np.ndarray
            Array of cluster centroids (shape: n_clusters, n_elements).
        labels : np.ndarray
            Cluster labels for each sample.
        els_std_dev_per_cluster : list
            List of standard deviations for each element in each cluster.
        unused_compositions_list : list
            List of compositions excluded from clustering.
        
        Returns
        -------
        None
    
        Notes
        -----
        - The plot is saved as 'Clustering_plot.png' in the analysis directory.
        - Uses matplotlib for plotting (2D or 3D based on the number of elements).
        - candidate phases and centroids are annotated; standard deviation is shown as ellipses (2D) or ellipsoids (3D).
        """
        # Set font parameters
        plt.rcParams['font.family'] = 'Arial'
        fontsize = 14
        labelpad = 12
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
    
        # Define axis label suffix
        axis_label_add = ' (w%)' if self.clustering_cfg.features == cnst.W_FR_CL_FEAT else ' (at%)'
        ticks = np.arange(0, 1, 0.1)
        ticks_labels = [f"{x*100:.0f}" for x in ticks]
    
        # Create figure and axes
        fig = plt.figure(figsize=(6, 6))
        if len(elements) == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlabel(elements[2] + axis_label_add, labelpad=labelpad * 0.95)
            ax.set_zlim(0, 1)
            ax.set_zticks(ticks)
            ax.set_zticklabels(ticks_labels)
        else:
            ax = fig.add_subplot(111)
    
        # Plot clustered datapoints
        ax.scatter(*els_comps_list, c=labels, cmap='viridis', marker='o')
    
        # Plot centroids
        ax.scatter(*centroids.T, c='red', marker='x', s=100, label='Centroids')
    
        # Plot standard deviation ellipses or ellipsoids
        first_ellipse = True
        for centroid, stdevs in zip(centroids, els_std_dev_per_cluster):
            if ~np.any(np.isnan(stdevs)):
                if len(elements) == 3:  # 3D plot
                    x_c, y_c, z_c = centroid
                    rx, ry, rz = stdevs
    
                    # Create the ellipsoid
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    x = x_c + rx * np.outer(np.cos(u), np.sin(v))
                    y = y_c + ry * np.outer(np.sin(u), np.sin(v))
                    z = z_c + rz * np.outer(np.ones_like(u), np.cos(v))
    
                    # Plot the surface with transparency
                    ax.plot_surface(x, y, z, color='red', alpha=0.1, edgecolor='none')
    
                    if first_ellipse:
                        first_ellipse = False
                        ax.plot([], [], [], color='red', alpha=0.1, label='Stddev')
                else:  # 2D plot
                    x_c, y_c = centroid
                    rx, ry = stdevs
    
                    # Plot the ellipse with transparency
                    ellipse = patches.Ellipse((x_c, y_c), rx, ry, edgecolor='red', facecolor='red', linestyle='--', alpha=0.2)
                    if first_ellipse:
                        ellipse.set_label('Stddev')
                        first_ellipse = False
                    ax.add_patch(ellipse)
    
        # Plot unused compositions (discarded from clustering)
        if unused_compositions_list and self.plot_cfg.show_unused_comps_clust:
            ax.scatter(*np.array(unused_compositions_list).T, c='grey', marker='^', label='Discarded comps.')
    
        # Plot candidate phases
        if self.ref_formulae is not None:
            first_ref = True
            ref_phases_df = self.ref_phases_df[elements]
            for index, row in ref_phases_df.iterrows():
                label = 'Candidate phases' if first_ref else None
                ax.scatter(*row.values, c='blue', marker='*', s=100, label=label)
                ref_label = to_latex_formula(self.ref_formulae[index])
                ax.text(*row.values, ref_label, color='black', fontsize=fontsize, ha='left', va='bottom')
                first_ref = False
    
        # Annotate centroids with their cluster labels
        for i, centroid in enumerate(centroids):
            ax.text(*centroid, str(i), color='black', fontsize=fontsize, ha='right', va='bottom')
    
        # Set axis labels and limits
        ax.set_xlabel(elements[0] + axis_label_add, labelpad=labelpad)
        ax.set_ylabel(elements[1] + axis_label_add, labelpad=labelpad)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks_labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks_labels)
        ax.set_title(f'{self.clustering_cfg.method} clustering {self.sample_cfg.ID}')
        
        if getattr(self.plot_cfg, 'show_legend_clustering', None):
            ax.legend(fontsize = fontsize)
        
        # plt.ion()
        if self.plot_cfg.show_plots:
            plt.show()
        # plt.pause(0.001)
        fig.savefig(os.path.join(self.analysis_dir, cnst.CLUSTERING_PLOT_FILENAME + cnst.CLUSTERING_PLOT_FILEEXT))
        # plt.close(fig)

    
    def _save_violin_plot_powder_mixture(
        self,
        W_mol_frs: List[float],
        ref_names: List[str],
        cluster_ID : int
    ) -> None:
        """
        Generate and save a violin plot visualizing the distribution of precursor molar fractions in a binary powder mixture.
    
        The plot displays:
          - The kernel density estimate (KDE) of the measured molar fractions
          - Individual measured values
          - The mean and standard deviation of the distribution
    
        Note:
            Prior to plotting, the precursor molar fractions are normalized so their sum equals 1.
            As a result, the standard deviation is identical for both precursors in the mixture.
    
        Parameters
        ----------
        W_mol_frs : list(float)
            Measured molar fractions of the precursors for the current cluster, represented as a binary mixture of two powders.
        ref_names : list(str)
            Chemical formulas (or names) of the two parent phases forming the mixture.
        cluster_ID : int
            Current cluster ID. Used for violin plot name
        """
    
        # --- Plot styling ---
        plt.rcParams['font.family'] = 'Arial'
        fontsize = 17
        labelpad = 0
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
        purple_cmap = cm.get_cmap('Purples')
        yellow_cmap = cm.get_cmap('autumn')
    
        # Extract coordinates from W
        y_vals = W_mol_frs[:, 0]
    
        fig, ax_left = plt.subplots(figsize=(4, 4))
        
        mean = np.mean(y_vals)
        std = np.std(y_vals)
        
        # Violin plot (default zorder is 1)
        ax_left = sns.violinplot(data=y_vals, inner=None, color=purple_cmap(0.3),
                                 linewidth=1.5, density_norm='area', width=1, zorder=1)
        
        # Swarmplot (zorder 2)
        sns.swarmplot(data=y_vals, color=purple_cmap(0.8),
                      edgecolor=purple_cmap(1.0), linewidth=2, size=5, label='data', zorder=2)
        
        # Error bars (zorder 3 and 4)
        ax_left.errorbar(0, mean, yerr=std / 2, fmt='none', color=yellow_cmap(0.9),
                         label='Mean ±1 Std Dev', capsize=5, elinewidth=1,
                         zorder=4, markerfacecolor=yellow_cmap(0.9),
                         markeredgecolor='black', markeredgewidth=1,
                         marker='o', linestyle='none')
        ax_left.errorbar(0, mean, yerr=std / 2, fmt='none', color='none',
                         label='_nolegend_', capsize=6, elinewidth=2,
                         zorder=3, markerfacecolor='none',
                         markeredgecolor='black', markeredgewidth=2,
                         marker='o', linestyle='none', ecolor='black')
        
        # Mean point (highest zorder, plotted last)
        ax_left.scatter(0, mean, color=yellow_cmap(0.9), marker='o', s=50,
                        edgecolors='k', linewidths=1, label='Mean', zorder=10)
    
        ax_left = plt.gca()
        ax_left.set_xticks([])
        ax_left.set_yticks([0, 1])  # Show ticks at 0 and 1 on left y-axis
        ax_left.set_frame_on(True)
        for spine in ax_left.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.5)
        plt.grid(False)
    
        plt.xlim(-0.5, 0.5)
        ylim_bottom = 0
        ylim_top = 1
        ax_left.set_ylim(ylim_bottom, ylim_top)
    
        # Left y-axis label (0→1)
        left_formula = to_latex_formula(ref_names[0], include_dollar_signs=False)
        ax_left.set_ylabel(rf"$x_{{\mathrm{{{left_formula}}}}}$", labelpad=labelpad)
        
        # Right y-axis (inverted 1→0)
        ax_right = ax_left.twinx()
        ax_right.set_ylim(ylim_top, ylim_bottom)
        ax_right.set_yticks([1, 0])  # Inverted ticks on right y-axis
        right_formula = to_latex_formula(ref_names[1], include_dollar_signs=False)
        ax_right.set_ylabel(rf"$x_{{\mathrm{{{right_formula}}}}}$", labelpad=labelpad)
    
        # Add standard deviation inside the plot
        ax_left.text(0.03, 0.03, rf"$\sigma_x = {std*100:.1f}$%", fontsize=fontsize,
                     color='black', ha='left', va='bottom', transform=ax_left.transAxes)
    
        ax_left.set_title(f'Violin plot {self.sample_cfg.ID}')
    
        # Save figure
        fig.savefig(
            os.path.join(self.analysis_dir,
                         cnst.POWDER_MIXTURE_PLOT_FILENAME + f"_cl{cluster_ID}_{ref_names[0]}_{ref_names[1]}" + cnst.CLUSTERING_PLOT_FILEEXT),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
        )

    @staticmethod
    def _save_silhouette_plot(
        model: 'KMeans',
        compositions_df: 'pd.DataFrame',
        results_dir: str,
        show_plot: bool
    ) -> None:
        """
        Generate and save a silhouette plot for the clustering results.
    
        Parameters
        ----------
        model : KMeans
            Fitted clustering model.
        compositions_df : pd.DataFrame
            DataFrame of sample compositions used for clustering.
        results_dir : str
            Directory where the plot will be saved.
        show_plot : bool
            If True, the plot will be displayed interactively.
    
        Returns
        -------
        None
    
        Notes
        -----
        - Uses Yellowbrick's SilhouetteVisualizer for plotting.
        - Suppresses harmless sklearn warnings during visualization.
        - The plot is saved as 'Silhouette_plot.png' in the results directory.
        """
        plt.figure(figsize=(10, 8))
        sil_visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        with warnings.catch_warnings():
            # Suppress harmless sklearn warnings
            warnings.simplefilter("ignore", UserWarning)
            sil_visualizer.fit(compositions_df)  # Fit the data to the visualizer
    
        plt.ylabel('Cluster label')
        plt.xlabel('Silhouette coefficient values')
        plt.legend(loc='upper right', frameon=True)
    
        if show_plot:
            plt.ion()
            sil_visualizer.show()
            plt.pause(0.001)
            plt.ioff()
    
        fig = sil_visualizer.fig
        fig.savefig(os.path.join(results_dir, 'Silhouette_plot.png'))
    
        # Close the figure if not displaying
        if not show_plot:
            plt.close(fig)
    
    
    #%% Save Data
    # =============================================================================
    def _save_result_and_stats(
        self,
        centroids: 'np.ndarray',
        els_std_dev_per_cluster: list,
        centroids_other_fr: 'np.ndarray',
        els_std_dev_per_cluster_other_fr: list,
        n_points_per_cluster: list,
        wcss_per_cluster: list,
        rms_dist_cluster: list,
        rms_dist_cluster_other_fr: list,
        refs_assigned_df: 'pd.DataFrame',
        wcss: float,
        sil_score: float,
        tot_n_points: int,
        max_analytical_error: float,
        clusters_assigned_mixtures: list
    ) -> None:
        """
        Save and store clustering results and statistics, including centroids, standard deviations, reference assignments, mixture assignments, and summary statistics.
    
        This method:
          - Constructs a DataFrame of cluster statistics and assignments.
          - Adds candidate phase and mixture assignments if available.
          - Saves the DataFrame to CSV and stores it as an attribute.
          - Saves general clustering information to a JSON file and stores it as an attribute.
    
        Parameters
        ----------
        centroids : np.ndarray
            Array of cluster centroids (shape: n_clusters, n_features).
        els_std_dev_per_cluster : list
            Standard deviations for each elemental fraction in each cluster (same shape as centroids).
        centroids_other_fr : np.ndarray
            Centroids in the alternate fraction representation.
        els_std_dev_per_cluster_other_fr : list
            Standard deviations of elemental fractions in the alternate fraction representation.
        n_points_per_cluster : list
            Number of points in each cluster.
        wcss_per_cluster : list
            Within-cluster sum of squares for each cluster.
        rms_dist_cluster : list
            Standard deviation of distances to centroid for each cluster.
        rms_dist_cluster_other_fr : list
            Standard deviation of distances in the alternate fraction representation.
        refs_assigned_df : pd.DataFrame
            DataFrame of reference assignments for each cluster.
        wcss : float
            Total within-cluster sum of squares.
        sil_score : float
            Silhouette score for the clustering.
        tot_n_points : int
            Total number of spectra considered.
        max_analytical_error : float
            Maximum allowed analytical error for spectra used in clustering.
        clusters_assigned_mixtures : list
            List of mixture assignments for each cluster.
    
        Returns
        -------
        None
    
        Notes
        -----
                - The cluster DataFrame is saved as a transposed '<cnst.CLUSTERS_FILENAME>.csv' in the analysis directory
                    for easier manual reading.
        - General clustering info is stored as an attribute for later use and persisted in the ledger.
    
        Raises
        ------
        OSError
            If the analysis directory cannot be created or files cannot be written.
    
        Suggestions
        -----------
        - Consider using more explicit type hints for lists, e.g., List[float] or List[int], and for DataFrames, use pd.DataFrame directly.
        - If you expect large data, consider saving DataFrames in a binary format (e.g., Parquet) for efficiency.
        """
    
        # Prepare cluster statistics as dictionaries for DataFrame construction
        els_fr = np.transpose(centroids)
        els_other_fr = np.transpose(centroids_other_fr)
        els_stdevs = np.transpose(els_std_dev_per_cluster)
        els_stdevs_other_fr = np.transpose(els_std_dev_per_cluster_other_fr)
    
        # Select keys for fraction and standard deviation columns based on configuration
        if self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
            fr_key = cnst.AT_FR_DF_KEY
            other_fr_key = cnst.W_FR_DF_KEY
        elif self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
            fr_key = cnst.W_FR_DF_KEY
            other_fr_key = cnst.AT_FR_DF_KEY
        else:
            # Suggestion: handle unexpected feature settings
            raise ValueError(f"Unknown clustering feature: {self.clustering_cfg.features}")
    
        stddev_key = cnst.STDEV_DF_KEY + fr_key
        other_stddev_key = cnst.STDEV_DF_KEY + other_fr_key
    
        # Prepare dictionaries for DataFrame columns
        els_fr_dict = {el + fr_key: np.round(el_comps * 100, 2) for el, el_comps in zip(self.all_els_sample, els_fr)}
        els_fr_stdevs_dict = {el + stddev_key: np.round(el_stddev * 100, 2) for el, el_stddev in zip(self.all_els_sample, els_stdevs)}
        els_other_fr_dict = {el + other_fr_key: np.round(el_comps * 100, 2) for el, el_comps in zip(self.all_els_sample, els_other_fr)}
        els_other_fr_stdevs_dict = {el + other_stddev_key: np.round(el_stddev * 100, 2) for el, el_stddev in zip(self.all_els_sample, els_stdevs_other_fr)}
    
        # Compose main cluster DataFrame
        clusters_dict = {
            cnst.N_PTS_DF_KEY: n_points_per_cluster,
            **els_fr_dict,
            **els_fr_stdevs_dict,
            **els_other_fr_dict,
            **els_other_fr_stdevs_dict,
            cnst.RMS_DIST_DF_KEY + fr_key: (np.array(rms_dist_cluster) * 100).round(2),
            cnst.RMS_DIST_DF_KEY + other_fr_key: (np.array(rms_dist_cluster_other_fr) * 100).round(2),
            cnst.WCSS_DF_KEY + fr_key: (np.array(wcss_per_cluster) * 10000).round(2)
        }
        clusters_df = pd.DataFrame(clusters_dict)
    
        # Add reference assignments if available
        if self.ref_formulae:
            clusters_df = pd.concat([clusters_df.reset_index(drop=True), refs_assigned_df.reset_index(drop=True)], axis=1)
    
        # Add mixture assignments if any
        mixtures_df = self._build_mixtures_df(clusters_assigned_mixtures)
        clusters_df = pd.concat([clusters_df.reset_index(drop=True), mixtures_df.reset_index(drop=True)], axis=1)
    
        # Ensure the analysis directory exists
        try:
            os.makedirs(self.analysis_dir, exist_ok=True)
        except Exception as e:
            raise OSError(f"Could not create analysis directory '{self.analysis_dir}': {e}")
    
        # Save and store DataFrame
        clusters_csv_path = os.path.join(self.analysis_dir, cnst.CLUSTERS_FILENAME + '.csv')
        try:
            clusters_csv_df = clusters_df.transpose().copy()
            clusters_csv_df.columns = [f"Cluster_{i}" for i in range(len(clusters_df))]
            clusters_csv_df.to_csv(clusters_csv_path, index=True, header=True, index_label="Metric")
        except Exception as e:
            raise OSError(f"Could not write clusters DataFrame to '{clusters_csv_path}': {e}")
    
        self.clusters_df = clusters_df
    
        # Save general clustering info and store for printing
        now = datetime.now()

        def _serialize_config(config_obj: Any) -> Dict[str, Any]:
            if hasattr(config_obj, "model_dump"):
                return config_obj.model_dump(mode="json")
            return asdict(config_obj)
    
        # Gather configuration dataclasses as dictionaries
        cfg_dataclasses = {
            cnst.QUANTIFICATION_CFG_KEY: _serialize_config(self.quant_cfg),
            cnst.CLUSTERING_CFG_KEY: _serialize_config(self.clustering_cfg),
            cnst.PLOT_CFG_KEY: _serialize_config(self.plot_cfg),
        }
    
        clustering_info = {
            cnst.DATETIME_KEY: now.strftime("%Y-%m-%d %H:%M:%S"),
            cnst.N_SP_ACQUIRED_KEY: tot_n_points,
            cnst.N_SP_USED_KEY: sum(n_points_per_cluster),
            cnst.N_CLUST_KEY: len(centroids),
            cnst.WCSS_KEY: wcss,
            cnst.SIL_SCORE_KEY: sil_score,
            **cfg_dataclasses,
        }
        self.clustering_info = clustering_info
        
        
    def _save_collected_data(
        self,
        labels: List,
        df_indices: List,
        backup_previous_data: Optional[bool] = True,
        include_spectral_data: Optional[bool] = True,
    ) -> None:
        """
        Save the collected spectra, their quantification (optionally), and their cluster assignments.
    
        This method builds a DataFrame with quantification and clustering information for each spectrum,
        along with spectral data if requested. It ensures unique output files and backs up existing data.
        Spectra with insufficient counts are handled gracefully.
    
        Parameters
        ----------
        labels : List
            List of cluster labels assigned to each spectrum (by index in df_indices).
        df_indices : List
            List of indices mapping labels to DataFrame rows.
        backup_previous_data : bool, optional
            Backs up previous data file if present (Default = True).
        include_spectral_data : bool, optional
            If True, includes raw spectral and background data in the output (default: True).
            
        Returns
        -------
        data_df: pd.Dataframe
            Dataframe object containing the saved data. Used only when measuring experimental standards
            None if no spectrum to save was acquired
    
        Notes
        -----
        - If a file with the intended name exists and contains quantification data, a counter is appended to the filename.
        - If include_spectral_data is False, only compositions are saved (to the analysis directory).
        - Existing main data files are backed up before being overwritten.
        - Uses make_unique_dir for unique directory creation if necessary.
    
        Raises
        ------
        OSError
            If the output directory cannot be created or files cannot be written.
        """
    
        is_standards_measurements = self.exp_stds_cfg.is_exp_std_measurement
    
        # Determine the number of spectra to process
        if include_spectral_data:
            n_spectra = len(self.spectral_data[cnst.SPECTRUM_DF_KEY])
        else:
            n_spectra = len(self.spectra_quant_records)
    
        if n_spectra == 0:
            return None
        
        data_list = []
        ledger_entries = []  # list of per-spectrum dicts for ledger.json
        for i in range(n_spectra):
            # Retrieve the typed QuantificationResult for this spectrum (None when not quantified)
            record = self.spectra_quant_records[i] if i < len(self.spectra_quant_records) else None
            has_result = record is not None and record.composition_atomic_fractions is not None

            if has_result:
                if is_standards_measurements:
                    exp_std_comp_d = self.exp_stds_cfg.w_frs
                    std_els = list(exp_std_comp_d.keys())
                    std_w_frs = list(exp_std_comp_d.values())
                    std_at_frs = weight_to_atomic_fr(std_w_frs, std_els, verbose=False)
                    atomic_comp = {el + cnst.AT_FR_DF_KEY: round(fr * 100, 2) for el, fr in zip(std_els, std_at_frs)}
                    weight_comp = {el + cnst.W_FR_DF_KEY: round(fr * 100, 2) for el, fr in exp_std_comp_d.items()}
                    meas_data = {
                        **atomic_comp,
                        **weight_comp,
                        cnst.COMP_AT_FR_KEY: dict(record.composition_atomic_fractions),
                        cnst.COMP_W_FR_KEY: dict(record.composition_weight_fractions),
                        cnst.AN_ER_KEY: float(record.analytical_error),
                    }
                    if record.fit_result is not None:
                        if record.fit_result.r_squared is not None:
                            meas_data[cnst.R_SQ_KEY] = float(record.fit_result.r_squared)
                        if record.fit_result.reduced_chi_squared is not None:
                            meas_data[cnst.REDCHI_SQ_KEY] = float(record.fit_result.reduced_chi_squared)
                else:
                    # Unpack spectral quantification results and convert from elemental fraction to % for readability
                    atomic_comp = {el + cnst.AT_FR_DF_KEY: round(fr * 100, 2) for el, fr in record.composition_atomic_fractions.items()}
                    weight_comp = {el + cnst.W_FR_DF_KEY: round(fr * 100, 2) for el, fr in record.composition_weight_fractions.items()}
                    analytical_er = {cnst.AN_ER_DF_KEY: round(float(record.analytical_error) * 100, 2)}

                    # Fit quality metrics (present only when a full fit was performed)
                    r_squared = None
                    redchi_sq = None
                    if record.fit_result is not None:
                        r_squared = record.fit_result.r_squared
                        redchi_sq = record.fit_result.reduced_chi_squared
        
                    # Extract cluster label if assigned
                    try:
                        label_index = df_indices.index(i)
                        cluster_n = labels[label_index]
                    except ValueError:
                        cluster_n = pd.NA
                    except AttributeError:
                        cluster_n = pd.NA
                    
                    # Compose row of data to be saved
                    meas_data = {
                        cnst.CL_ID_DF_KEY: cluster_n,
                        **atomic_comp,
                        **analytical_er,
                        **weight_comp,
                        cnst.R_SQ_KEY: r_squared,
                        cnst.REDCHI_SQ_KEY: redchi_sq,
                    }
                    
                # Compose row of data to be saved
                data_row = {
                    **self.sp_coords[i],
                    **meas_data
                }
            else:
                # Counts in this spectrum were too low or quantification was interrupted
                data_row = self.sp_coords[i]
            
            # Add comment and quantification flag columns, if available
            try:
                data_row[cnst.COMMENTS_DF_KEY] = self.spectral_data[cnst.COMMENTS_DF_KEY][i]
                data_row[cnst.QUANT_FLAG_DF_KEY] = self.spectral_data[cnst.QUANT_FLAG_DF_KEY][i]
            except Exception:
                pass
    
            # Add spectral data
            if include_spectral_data:
                try:  # If background present
                    background_entry = '[' + ','.join(map(str, self.spectral_data[cnst.BACKGROUND_DF_KEY][i])) + ']'
                except Exception:
                    background_entry = None  # Use None so pandas will recognize as missing
    
                real_time = self.spectral_data[cnst.REAL_TIME_DF_KEY][i]
                live_time = self.spectral_data[cnst.LIVE_TIME_DF_KEY][i]
    
                if real_time is not None:
                    real_time = round(real_time, 2)
                if live_time is not None:
                    live_time = round(live_time, 2)
    
                # Format strings to avoid truncation when saving dataframe into csv
                data_row = {
                    **data_row,
                    cnst.REAL_TIME_DF_KEY: real_time,
                    cnst.LIVE_TIME_DF_KEY: live_time,
                    cnst.SPECTRUM_DF_KEY: '[' + ','.join(map(str, self.spectral_data[cnst.SPECTRUM_DF_KEY][i])) + ']',
                    cnst.BACKGROUND_DF_KEY: background_entry
                }
    
            data_list.append(data_row)

            # Build ledger entry for ledger.json (spectral data + coordinates)
            if include_spectral_data:
                try:
                    bg_vals = list(self.spectral_data[cnst.BACKGROUND_DF_KEY][i])
                except Exception:
                    bg_vals = None
                ledger_entry = {
                    **self.sp_coords[i],
                    cnst.REAL_TIME_DF_KEY: real_time,
                    cnst.LIVE_TIME_DF_KEY: live_time,
                    cnst.SPECTRUM_DF_KEY: list(self.spectral_data[cnst.SPECTRUM_DF_KEY][i]),
                    cnst.BACKGROUND_DF_KEY: bg_vals,
                }
                ledger_entries.append(ledger_entry)
    
        # Convert list of dictionaries to DataFrame
        data_df = pd.DataFrame(data_list)
    
        # Remove Cluster ID column if no value has been assigned
        if cnst.CL_ID_DF_KEY in data_df.columns:
            if data_df[cnst.CL_ID_DF_KEY].isna().all():
                data_df.pop(cnst.CL_ID_DF_KEY)
            else:
                # Convert to nullable integer Int64 dtype
                data_df[cnst.CL_ID_DF_KEY] = data_df[cnst.CL_ID_DF_KEY].astype('Int64')
    
        # Remove background column if it is entirely None or NaN
        if cnst.BACKGROUND_DF_KEY in data_df.columns:
            if data_df[cnst.BACKGROUND_DF_KEY].isna().all():
                data_df.pop(cnst.BACKGROUND_DF_KEY)
    
        # Reorder columns to ensure spectral data is at the end
        columns = data_df.columns.tolist()
        last_columns = [
            cnst.R_SQ_KEY, cnst.REDCHI_SQ_KEY,
            cnst.QUANT_FLAG_DF_KEY, cnst.COMMENTS_DF_KEY,
            cnst.REAL_TIME_DF_KEY, cnst.LIVE_TIME_DF_KEY,
            cnst.SPECTRUM_DF_KEY, cnst.BACKGROUND_DF_KEY
        ]
        remaining_columns = [col for col in columns if col not in last_columns]
        new_column_order = remaining_columns + [col for col in last_columns if col in columns]
        data_df = data_df[new_column_order]

        # Build CSV-only DataFrame: drop spectral arrays and spatial coordinates (kept in ledger.json)
        csv_exclude_columns = [
            cnst.SPECTRUM_DF_KEY, cnst.BACKGROUND_DF_KEY,
            cnst.REAL_TIME_DF_KEY, cnst.LIVE_TIME_DF_KEY,
            cnst.SP_X_COORD_DF_KEY, cnst.SP_Y_COORD_DF_KEY,
            cnst.SP_X_PIXEL_COORD_DF_KEY, cnst.SP_Y_PIXEL_COORD_DF_KEY,
        ]
        csv_df = data_df.drop(columns=[c for c in csv_exclude_columns if c in data_df.columns])

        # Save dataframe
        if data_df is not None and include_spectral_data:
            # Keep standards measurement exports unchanged, but avoid writing Data.csv for sample quantification runs.
            if is_standards_measurements:
                base_name = f'{cnst.STDS_MEAS_FILENAME}'
                legacy_base_name = f'{base_name}_legacy'
                extension = f'{cnst.DATA_FILEEXT}'
                data_path = os.path.join(self.sample_result_dir, base_name + extension)

                if os.path.exists(data_path) and backup_previous_data:
                    if cnst.QUANT_FLAG_DF_KEY in pd.read_csv(data_path, nrows=0).columns:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        backup_path = make_unique_path(self.sample_result_dir, f'{base_name}_old_{timestamp}', extension)
                        try:
                            shutil.copyfile(data_path, backup_path)
                        except Exception as e:
                            backup_successful = False
                            raise OSError(f"Could not backup previous {data_path} file. This file was not overwritten."
                                          f"Ensure you ovewrite it with a copy of {base_name}{extension} prior analysis:"
                                          f"{e}")
                        else:
                            backup_successful = True
                    else:
                        backup_successful = True
                else:
                    backup_successful = True

                try:
                    if backup_successful:
                        if isinstance(self.output_filename_suffix, str) and self.output_filename_suffix != '':
                            if backup_previous_data:
                                data_path_with_suffix = make_unique_path(self.sample_result_dir, base_name + self.output_filename_suffix, extension)
                                legacy_path_with_suffix = make_unique_path(self.sample_result_dir, legacy_base_name + self.output_filename_suffix, extension)
                            else:
                                data_path_with_suffix = os.path.join(self.sample_result_dir, base_name + self.output_filename_suffix + extension)
                                legacy_path_with_suffix = os.path.join(self.sample_result_dir, legacy_base_name + self.output_filename_suffix + extension)
                            csv_df.to_csv(data_path_with_suffix, index=False, header=True)
                            data_df.to_csv(legacy_path_with_suffix, index=False, header=True)
                        else:
                            csv_df.to_csv(data_path, index=False, header=True)
                            legacy_data_path = os.path.join(self.sample_result_dir, legacy_base_name + extension)
                            data_df.to_csv(legacy_data_path, index=False, header=True)
                    else:
                        csv_df.to_csv(data_path_with_suffix, index=False, header=True)
                        data_df.to_csv(legacy_path_with_suffix, index=False, header=True)
                except Exception as e:
                    raise OSError(f"Could not write '{data_path}': {e}")

            # Write ledger.json with per-spectrum spectral data and spatial coordinates
            if ledger_entries:
                ledger_path = os.path.join(
                    self.sample_result_dir,
                    cnst.LEDGER_FILENAME + cnst.LEDGER_FILEEXT
                )
                try:
                    existing_ledger = self._load_or_create_ledger()

                    spectra = []
                    for i, entry in enumerate(ledger_entries):
                        raw_x = entry.get(cnst.SP_X_COORD_DF_KEY, '')
                        raw_y = entry.get(cnst.SP_Y_COORD_DF_KEY, '')
                        raw_x_pixel = entry.get(cnst.SP_X_PIXEL_COORD_DF_KEY, '')
                        raw_y_pixel = entry.get(cnst.SP_Y_PIXEL_COORD_DF_KEY, '')

                        acquisition_details = AcquisitionDetails(
                            frame_id=str(entry.get(cnst.FRAME_ID_DF_KEY, '')).strip() or None,
                            particle_id=self._parse_optional_int(entry.get(cnst.PAR_ID_DF_KEY, '')),
                            spot_coordinates=self._build_spot_coordinates(raw_x, raw_y, raw_x_pixel, raw_y_pixel),
                        )

                        existing_results = []
                        if existing_ledger is not None and i < len(existing_ledger.spectra):
                            existing_results = list(existing_ledger.spectra[i].quantification_results)
                        spectrum_id = str(entry.get(cnst.SP_ID_DF_KEY, ''))
                        spectrum_vals = list(entry[cnst.SPECTRUM_DF_KEY])
                        spectrum_id_resolved = spectrum_id if spectrum_id else str(i)
                        live_time = self._coerce_optional_finite_float(entry.get(cnst.LIVE_TIME_DF_KEY))
                        real_time = self._coerce_optional_finite_float(entry.get(cnst.REAL_TIME_DF_KEY))
                        spectrum_relpath = self._resolve_or_create_spectrum_pointer(
                            spectrum_id=spectrum_id_resolved,
                            spectrum_vals=spectrum_vals,
                            live_time=live_time,
                            real_time=real_time,
                        )
                        background_relpath = None
                        background_vals = list(entry[cnst.BACKGROUND_DF_KEY]) if entry.get(cnst.BACKGROUND_DF_KEY) is not None else None
                        if background_vals is not None:
                            background_relpath = self._write_manufacturer_background_vector(
                                spectrum_id=spectrum_id_resolved,
                                background_vals=background_vals,
                            )
                        spectra.append(
                            SpectrumEntry(
                                live_acquisition_time=live_time if live_time is not None else (real_time if real_time is not None else 1.0),
                                total_counts=int(round(float(np.sum(spectrum_vals)))),
                                spectrum_id=spectrum_id_resolved,
                                spectrum_relpath=spectrum_relpath,
                                instrument_background_relpath=background_relpath,
                                acquisition_details=acquisition_details,
                                quantification_results=existing_results,
                            )
                        )

                    existing_configs = []
                    if existing_ledger is not None:
                        existing_configs = list(existing_ledger.quantification_configs)

                    ledger = SampleLedger(
                        sample_id=self.sample_cfg.ID,
                        sample_path=os.path.abspath(self.sample_result_dir),
                        configs=(
                            existing_ledger.configs
                            if existing_ledger is not None and existing_ledger.configs is not None
                            else self._build_ledger_configs()
                        ),
                        spectra=spectra,
                        quantification_configs=existing_configs,
                        active_quant=(
                            existing_ledger.active_quant
                            if existing_ledger is not None
                            else (existing_configs[-1].quantification_id if existing_configs else None)
                        ),
                    )

                    has_quant_records = any(
                        record is not None for record in getattr(self, 'spectra_quant_records', [])
                    )
                    if has_quant_records and getattr(self, 'current_quant_config', None) is not None:
                        self._upsert_current_quantification_config_on_ledger(ledger)
                        ledger.active_quant = self.current_quant_config.quantification_id

                        for i, quant_record in enumerate(getattr(self, 'spectra_quant_records', [])):
                            if quant_record is None or i >= len(ledger.spectra):
                                continue
                            existing_ids = {
                                existing.quantification_id
                                for existing in ledger.spectra[i].quantification_results
                            }
                            if quant_record.quantification_id not in existing_ids:
                                ledger.append_quantification_result(i, quant_record)

                    ledger.to_json_file(ledger_path)
                except Exception as e:
                    raise OSError(f"Could not write ledger to '{ledger_path}': {e}")

        else:
            # Save only compositions in Compositions.csv.
            # Used when cluster analysis is performed and cluster_IDs are assigned to each spectrum
            comp_path = os.path.join(self.analysis_dir, cnst.COMPOSITIONS_FILENAME + '.csv')
            compositions_df = data_df.drop(
                columns=[
                    c for c in [
                        cnst.SP_X_COORD_DF_KEY,
                        cnst.SP_Y_COORD_DF_KEY,
                        cnst.SP_X_PIXEL_COORD_DF_KEY,
                        cnst.SP_Y_PIXEL_COORD_DF_KEY,
                    ]
                    if c in data_df.columns
                ]
            )
            try:
                compositions_df.to_csv(comp_path, index=False, header=True)
            except Exception as e:
                raise OSError(f"Could not write compositions data to '{comp_path}': {e}")

        return data_df


    def _save_experimental_config(self, is_XSp_measurement) -> None:
        """
        Save all relevant configuration dataclasses and metadata related to the
        current spectrum collection/acquisition to a JSON file.
    
        The saved file includes:
            - Timestamp of saving
            - All configuration dataclasses
    
        This function is intended to be called after acquisition to ensure
        reproducibility and traceability of the experimental configuration.
    
        Raises
        ------
        OSError
            If the output directory cannot be created or file cannot be written.
        """
    
        now = datetime.now()

        def _serialize_config(config_obj: Any) -> Dict[str, Any]:
            if hasattr(config_obj, "model_dump"):
                return config_obj.model_dump(mode="json")
            return asdict(config_obj)
    
        # Gather configuration dataclasses as dictionaries
        cfg_dataclasses = {
            cnst.SAMPLE_CFG_KEY: _serialize_config(self.sample_cfg),
            cnst.MICROSCOPE_CFG_KEY: _serialize_config(self.microscope_cfg),
            cnst.MEASUREMENT_CFG_KEY: _serialize_config(self.measurement_cfg),
            cnst.SAMPLESUBSTRATE_CFG_KEY: _serialize_config(self.sample_substrate_cfg),
        }
        
        if is_XSp_measurement:
            cfg_dataclasses[cnst.QUANTIFICATION_CFG_KEY] = _serialize_config(self.quant_cfg)
    
        # Include dataclass corresponding to sample type
        if self.sample_cfg.is_powder_sample:
            cfg_dataclasses[cnst.POWDER_MEASUREMENT_CFG_KEY] = _serialize_config(self.powder_meas_cfg)
        elif self.sample_cfg.is_grid_acquisition:
            cfg_dataclasses[cnst.BULK_MEASUREMENT_CFG_KEY] = _serialize_config(self.bulk_meas_cfg)
        
        # Include dataclasses corresponding to measurement type
        if self.exp_stds_cfg.is_exp_std_measurement:
            cfg_dataclasses[cnst.EXP_STD_MEASUREMENT_CFG_KEY] = _serialize_config(self.exp_stds_cfg)
        elif is_XSp_measurement:
            cfg_dataclasses[cnst.CLUSTERING_CFG_KEY] = self._runtime_clustering_cfg_payload(self.clustering_cfg)
            cfg_dataclasses[cnst.PLOT_CFG_KEY] = _serialize_config(self.plot_cfg)


    
        # Compose the metadata dictionary for saving
        spectrum_collection_info = {
            cnst.DATETIME_KEY: now.strftime("%Y-%m-%d %H:%M:%S"),
            **cfg_dataclasses
        }
    
        # Ensure the output directory exists before saving
        try:
            os.makedirs(self.sample_result_dir, exist_ok=True)
        except Exception as e:
            raise OSError(f"Could not create output directory '{self.sample_result_dir}': {e}")
    
        try:
            existing_ledger = self._load_existing_ledger()
            ledger = self._build_ledger_from_current_state(existing_ledger=existing_ledger)
            ledger.configs = LedgerConfigs.model_validate({
                cnst.MICROSCOPE_CFG_KEY: spectrum_collection_info[cnst.MICROSCOPE_CFG_KEY],
                cnst.SAMPLE_CFG_KEY: spectrum_collection_info[cnst.SAMPLE_CFG_KEY],
                cnst.MEASUREMENT_CFG_KEY: spectrum_collection_info[cnst.MEASUREMENT_CFG_KEY],
                cnst.SAMPLESUBSTRATE_CFG_KEY: spectrum_collection_info[cnst.SAMPLESUBSTRATE_CFG_KEY],
                cnst.CLUSTERING_CFG_KEY: spectrum_collection_info.get(
                    cnst.CLUSTERING_CFG_KEY,
                    self._runtime_clustering_cfg_payload(self.clustering_cfg),
                ),
                cnst.PLOT_CFG_KEY: spectrum_collection_info.get(cnst.PLOT_CFG_KEY, self.plot_cfg.model_dump(mode="json")),
                cnst.POWDER_MEASUREMENT_CFG_KEY: spectrum_collection_info.get(cnst.POWDER_MEASUREMENT_CFG_KEY),
                cnst.BULK_MEASUREMENT_CFG_KEY: spectrum_collection_info.get(cnst.BULK_MEASUREMENT_CFG_KEY),
                cnst.EXP_STD_MEASUREMENT_CFG_KEY: spectrum_collection_info.get(cnst.EXP_STD_MEASUREMENT_CFG_KEY),
            })
            ledger.to_json_file(self._get_ledger_path())
        except Exception as e:
            raise OSError(f"Could not persist configurations to ledger '{self._get_ledger_path()}': {e}")
            
    #%% Print Results
    # =============================================================================            
    def _report_n_discarded_spectra(
        self,
        n_datapts: int,
        max_analytical_error: float
    ) -> None:
        """
        Print a summary report of discarded spectra and the reasons for their exclusion.
    
        Parameters
        ----------
        n_datapts : int
            Total number of spectra considered.
        max_analytical_error : float
            Maximum allowed analytical error (as a fraction, e.g., 0.05 for 5%). If None, this check is skipped.
    
        Notes
        -----
        - Prints detailed reasons for spectrum exclusion if self.verbose is True.
        - Advises user to check comments in the exported CSV files for more information.
        - Quantification flags indicate whether the quantification or the fit of each spectrum is likely to be affected by large errors:
            0: Quantification is ok, although it may be affected by large analytical error
           -1: As above, but quantification did not converge within 30 steps
            1: Error during EDS acquisition. No fit executed
            2: Total number of counts is lower than 95% of target counts, likely due to wrong segmentation. No fit executed
            3: Spectrum has too low signal in its low-energy portion, leading to poor quantification in this region. No fit executed
            4: Poor fit. Fit interrupted if interrupt_fits_bad_spectra=True
            5: Too high analytical error (>50%) indicating a missing element or other major sources of error. Fit interrupted if interrupt_fits_bad_spectra=True
            6: Excessive X-ray absorption. Fit interrupted if interrupt_fits_bad_spectra=True
            7: Excessive signal contamination from substrate
            8: Too few background counts below reference peak, likely leading to large quantification errors
            9: Unknown fitting error
        """
        is_any_spectrum_discarded = self.n_sp_too_low_counts + self.n_sp_bad_quant + self.n_sp_too_high_an_err > 0
        if not (self.verbose and n_datapts > 0 and is_any_spectrum_discarded):
            return
    
        print_single_separator()
        print("Summary of Discarded Spectra")
        print("  → For details, see the 'Comments' column in Compositions.csv.")
    
        # Discarded due to low counts, insufficient background, or acquisition/fitting errors
        if self.n_sp_too_low_counts > 0:
            print(
                f"  • {self.n_sp_too_low_counts} spectra were discarded due to insufficient total counts, "
                f"background counts below the threshold ({self.clustering_cfg.min_bckgrnd_cnts}), "
                "or errors during spectrum collection/fitting."
            )
    
        # Discarded due to quantification flags
        if self.n_sp_bad_quant > 0:
            print(
                f"  • {self.n_sp_bad_quant} spectra were discarded because they were flagged during quantification."
            )
    
        # Warning if more than half of the spectra were flagged
        if self.n_sp_bad_quant / n_datapts > 0.5:
            print_single_separator()
            print("  Warning: More than 50% of spectra were flagged during quantification!")
            print(
                "  Common causes for poor fits (quant_flag = 4) include missing elements in the fit.\n"
                "  Ensure that all elements present in your sample have been specified in the 'elements' argument "
                "  when initializing EMXSp_Composition_Analyzer."
            )
            
        # Discarded due to high analytical error
        if self.n_sp_too_high_an_err > 0:
            print(
                f"  • {self.n_sp_too_high_an_err} spectra were discarded because their analytical error "
                f"exceeded the maximum allowed value of {max_analytical_error*100:.1f}%."
            )
    
        print_single_separator()
            
        
    def print_results(self, n_cnd_to_print = 2, n_mix_to_print = 2) -> None:
        """
        Print a summary of clustering results, including clustering configuration, metrics,
        and a table of identified phases with elemental fractions, standard deviations,
        and reference/mixture assignments if present.
    
        The method:
          - Prints main clustering configuration and metrics.
          - Prints a table of phases, each with number of points, elemental fractions (with stddev),
            cluster stddev, WCSS, reference assignments, and mixture information if available.
        
        Parameters
        ----------
        n_cnd_to_print : int
            Max number of candidate phases and relative confidence scores to show. Candidates with scores
            close to 0 are not shown.
        n_mix_to_print : int
            Max number of candidate mixtures and relative confidence scores to show. Mixtures with scores
            close to 0 are not shown.
        
        Raises
        ------
        AttributeError
            If required attributes (clustering_info, clusters_df, etc.) are missing.
        KeyError
            If expected keys are missing from clustering_info or clusters_df.
        """
        # Print clustering info
        print_double_separator()
        print(f"Compositional analysis results for sample {self.sample_cfg.ID}:")
        print_single_separator()
        try:
            print('Clustering method: %s' % self.clustering_cfg.method)
            print('Clustering features: %s' % self.clustering_cfg.features)
            print('k finding method: %s' % self.clustering_cfg.k_finding_method)
            print('Number of clusters: %d' % self.clustering_info[cnst.N_CLUST_KEY])
            print('WCSS (%%): %.2f' % (self.clustering_info[cnst.WCSS_KEY] * 10000))
            print('Silhouette score: %.2f' % self.clustering_info[cnst.SIL_SCORE_KEY])
        except KeyError as e:
            raise KeyError(f"Missing key in clustering_info: {e}")
        except AttributeError as e:
            raise AttributeError(f"Missing attribute: {e}")
    
        # Print details on identified phases
        print_single_separator()
        print('Identified phases:')
        # Print stddev in-column for ease of visualization
        try:
            clusters_df = self.clusters_df
            el_fr_feature_key = cnst.AT_FR_DF_KEY if self.clustering_cfg.features == cnst.AT_FR_CL_FEAT else cnst.W_FR_DF_KEY
            fr_labels = [el + el_fr_feature_key for el in self.all_els_sample]
            stddev_labels = [el + cnst.STDEV_DF_KEY + el_fr_feature_key for el in self.all_els_sample]
            df_mod_to_print = []
            for index, row in clusters_df.iterrows():
                els_dict = {}
                df_mod_to_print.append({cnst.N_PTS_DF_KEY: row[cnst.N_PTS_DF_KEY]})
                # Add conversion to atomic fraction when mass fractions are used as features
                if self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
                    at_fr_dict = {}
                    for el in self.all_els_sample:
                        label = el + cnst.AT_FR_DF_KEY
                        at_fr_dict[label] = row[label]
                    df_mod_to_print[-1].update(at_fr_dict)
                # Get elemental fractions (at_fr or w_fr)
                for element, fr_l, stddev_l in zip(self.all_els_sample, fr_labels, stddev_labels):
                    els_dict[element + el_fr_feature_key] = f"{row[fr_l]:.1f} ± {row[stddev_l]:.1f}"
                # Add elemental fractions + cluster stddev and wcss
                df_mod_to_print[-1].update({
                    **els_dict,
                })
                
                # Add cluster-level std dev entries ---
                cluster_stddev_entries = {
                    col: f"{row[col]:.1f}" 
                    for col in clusters_df.columns 
                    if col.startswith(cnst.RMS_DIST_DF_KEY)
                }
                df_mod_to_print[-1].update(cluster_stddev_entries)
                
                # Add references if present
                if self.ref_formulae:
                    ref_keys_to_print = [key for i in range(1, n_cnd_to_print + 1) for key in [f'{cnst.CS_CND_DF_KEY}{i}', f'{cnst.CND_DF_KEY}{i}']]
                    ref_dict = {key: value for key, value in row.items() if key in ref_keys_to_print}
                    df_mod_to_print[-1].update(ref_dict)
                # Add mixtures to the printed report
                mix_keys_to_print = [key for i in range(1, n_mix_to_print + 1) for key in [f'{cnst.MIX_DF_KEY}{i}', f'{cnst.CS_MIX_DF_KEY}{i}']]
                mix_dict = {key: value for key, value in row.items() if key in mix_keys_to_print}
                df_mod_to_print[-1].update(mix_dict)
            # Set display options for float precision
            with pd.option_context('display.float_format', '{:,.2f}'.format):
                pd.set_option('display.max_columns', None)  # Display all columns
                print(pd.DataFrame(df_mod_to_print))
        except Exception as e:
            raise RuntimeError(f"Error printing phase results: {e}")
    

    #%% Experimental standard PB ratios
    # =============================================================================
    def _compile_standards_from_references(self) -> dict:
        """
        Compile a standards dictionary for the current sample by using the input
        candidate phases, if present in the list of standards.
    
        This function loads the standards library, iterates over all elements in
        the current sample, and for each X-ray reference line:
          - Verifies if the candidate phase compositions are present in the standards.
          - If no candidate phases are found, a warning is issued and existing
            standards are used.
          - If references are found, the function computes the mean of the
            corrected PB values and substitutes them into the standards dictionary.
    
        Returns:
            dict: The updated standards dictionary, where each entry for a given
                  element-line combination contains either existing standards or
                  a single mean standard to be fed to XSp_Quantifier
    
        Warns:
            UserWarning: If none of the input candidate phase compositions are
                    present for a given reference line in the standards file.
                    
        Note
        ----
        Currently only used when analysing mixtures of known powder precursors
        (i.e., powder_meas_cfg.is_known_powder_mixture_meas = True)
        """
        std_dict_all_modes, stds_filepath = self._load_xsp_standards()
        std_dict_all_lines = std_dict_all_modes[self.measurement_cfg.mode]
        ref_lines = XSp_Quantifier.xray_quant_ref_lines
        ref_formulae = self.clustering_cfg.ref_formulae
        
        filtered_std_dict = {}
        for el in self.detectable_els_sample:
            for line in ref_lines:
                el_line = f"{el}_{line}"
                if el_line not in std_dict_all_lines:
                    continue

                # Gather matching reference entries by comparing chemical formulas
                ref_entries = []
                for i, std_dict in enumerate(std_dict_all_lines[el_line]):
                    if std_dict[cnst.STD_ID_KEY] != cnst.STD_MEAN_ID_KEY:
                        try:
                            std_comp = Composition(std_dict[cnst.STD_FORMULA_KEY])
                            for ref_formula in ref_formulae:
                                if std_comp.reduced_formula == Composition(ref_formula).reduced_formula:
                                    ref_entries += [i]
                        except Exception as e:
                            warnings.warn(
                                f"Could not parse formula '{std_dict[cnst.STD_FORMULA_KEY]}' "
                                f"or compare with reference formulas {ref_formulae}. Error: {e}"
                            )
                    else:
                        std_mean_value = std_dict[cnst.COR_PB_DF_KEY]

                if len(ref_entries) < 1 and not self.exp_stds_cfg.is_exp_std_measurement:
                    text_line = "provided standards" if stds_filepath == "" else f"standards file at: {stds_filepath}"
                    warnings.warn(
                        f"None of the input candidate phases {ref_formulae} "
                        f"is present for line {el_line} in the {text_line}. "
                        "Using other available standards."
                    )
                    ref_value = std_mean_value # Mean value used for regular quantification
                else:
                    # Compute mean PB value from all available references
                    new_std_ref_list = [std_d for i, std_d in enumerate(std_dict_all_lines[el_line]) if i in ref_entries]
                    list_PB = [ref_line[cnst.COR_PB_DF_KEY] for ref_line in new_std_ref_list]
                    ref_value = float(np.mean(list_PB))
        
                std_dict_mean = {
                    cnst.STD_ID_KEY: cnst.STD_MEAN_ID_KEY,
                    cnst.COR_PB_DF_KEY: ref_value,
                }
                filtered_std_dict[el_line] = [std_dict_mean]

        return filtered_std_dict

    
    def _fit_stds_and_save_results(self, backup_previous_data: bool = True) -> Union[Tuple, None]:
        """
        Fit spectra collected from experimental standards, process results, and save them to disk.
    
        Parameters
        ----------
        backup_previous_data : bool, optional
            Backs up previous data file if present (Default = True).
    
        Returns
        -------
        Tuple or None
            If data was successfully processed:
                - std_ref_lines : Any
                    Data structure containing assembled standard PB data.
                - results_df : pandas.DataFrame
                    DataFrame containing averaged PB ratios and corrected values.
                - Z_sample : Any
                    Sample average atomic number computed with different methods.
            If no measurement data was available:
                Returns `(None, None, None)`.
    
        Raises
        ------
        RuntimeError
            If fitting, saving, or PB correction fails unexpectedly.
        """
        
        # Initialize return variables
        std_ref_lines = None
        results_df = None
        Z_sample = None
        
        try:
            # Fit spectra and assemble results
            self._fit_and_quantify_spectra(quantify=False)
        except Exception as e:
            raise RuntimeError(f"Error during fitting and quantification: {e}") from e
        
        try:
            # Save per-spectrum measurement results
            data_df = self._save_collected_data(
                None,
                None,
                backup_previous_data=backup_previous_data,
                include_spectral_data=True
            )
        except Exception as e:
            raise RuntimeError(f"Error while saving collected data: {e}") from e
        
        if data_df is not None and not data_df.empty:
            try:
                # Assemble PB data and calculate corrections
                std_ref_lines = self._assemble_std_PB_data(data_df)
                if std_ref_lines != {}:
                    PB_corrected, Z_sample = self._calc_corrected_PB(std_ref_lines)
                    
                    # Save averaged PB results
                    results_df = self._save_std_results(std_ref_lines, PB_corrected)
                else:
                    if self.verbose:
                        print("No valid standard measurement acquired.")
            except Exception as e:
                raise RuntimeError(f"Error while processing standard results: {e}") from e
    
            return std_ref_lines, results_df, Z_sample
        
        # No data available to process
        return None, None, None

    
    def _evaluate_exp_std_fit(self, tot_n_spectra: int) -> Tuple[bool, bool]:
        """
        Evaluate the experimental standard fitting results after collecting a given number of spectra.
    
        This method attempts to fit the experimental standards using the currently collected spectra.
        It determines whether the fit was successful and whether the minimum required number of valid
        spectra has been reached. The method also provides verbose feedback if enabled.
    
        Parameters
        ----------
        tot_n_spectra : int
            Total number of spectra collected so far.
    
        Returns
        -------
        Tuple[bool, bool]
            A Tuple containing:
            - is_fit_successful (bool): Whether the fitting process produced valid results.
            - is_converged (bool): Whether the minimum number of valid spectra was reached.
        """
        is_fit_successful = False
        is_converged = False
    
        try:
            if self.verbose:
                print_double_separator()
                print(f"Fitting after collection of {tot_n_spectra} spectra...")
    
            _, results_df, _ = self._fit_stds_and_save_results(backup_previous_data=False)
    
            if results_df is not None and not results_df.empty:
                is_fit_successful = True
    
                # Retrieve the minimum number of valid spectra from the results
                try:
                    num_valid_spectra = int(np.min(results_df[cnst.N_SP_USED_KEY]))
                except (KeyError, ValueError, TypeError) as e:
                    raise RuntimeError(f"Results DataFrame missing or invalid '{cnst.N_SP_USED_KEY}' column.") from e
    
                is_converged = num_valid_spectra >= self.min_n_spectra
    
                if self.verbose:
                    print_double_separator()
                    print("Fitting performed.")
                    print(f"{num_valid_spectra} valid spectra were collected.")
                    if is_converged:
                        print(f"Target number of {self.min_n_spectra} was reached.")
            else:
                if self.verbose:
                    print_double_separator()
                    print("No valid spectrum collected.")
    
            # If not converged, provide feedback
            if not is_converged and self.verbose:
                if tot_n_spectra >= self.max_n_spectra:
                    print(f"Maximum allowed number of {self.max_n_spectra} spectra was acquired, "
                          f"but target number of {self.min_n_spectra} was not reached.")
                else:
                    print(f"More spectra will be collected to reach target number of {self.min_n_spectra}.")
    
        except Exception as e:
            raise RuntimeError("An error occurred while evaluating the experimental standard fit.") from e
    
        return is_fit_successful, is_converged
    
        
    def _assemble_std_PB_data(
        self,
        data_df: "pd.DataFrame"
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Assemble Peak-to-Background (PB) ratio data for the experimental standard references to use during quantification.
    
        This method processes the provided DataFrame of spectral data to:
        1. Remove any X-ray peaks whose PB ratio is absent or below the acceptable threshold for all spectra.
        2. Exclude spectra that do not meet the accepted quantification flags.
        3. Compile PB ratio statistics (mean, std. dev) and corresponding theoretical energies for each relevant element line.
    
        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame containing PB ratio measurements and associated metadata.
            Must contain:
            - `cnst.QUANT_FLAG_DF_KEY` column for quantification flags.
            - Columns for PB ratios of element lines (e.g., 'Fe_Ka', 'Cu_Ka', etc.).
            - Possibly NaN values where peaks are absent.
    
        Returns
        -------
        Dict[str, Dict[str, float | List[float]]]
            Dictionary mapping each fitted standard element line to a sub-dictionary containing:
            - cnst.PB_RATIO_KEY: List of measured PB ratios.
            - cnst.MEAN_PB_KEY: mean PB ratio (ignoring NaN).
            - cnst.STDEV_PB_DF_KEY: standard deviation of PB ratios (ignoring NaN).
            - cnst.PEAK_TH_ENERGY_KEY: theoretical peak energy for that element line.
    
        Notes
        -----
        - Assumes that `self._th_peak_energies` is a dictionary mapping element lines to their theoretical energies.
        """
        # Filter out X-ray peaks whose PB ratio is absent for all spectra
        data_df = data_df.dropna(axis=1, how="all")
        # Filter out rows corresponding to spectra that should be discarded
        try:
            data_filtered_df = data_df[data_df[cnst.QUANT_FLAG_DF_KEY].isin(self.exp_stds_cfg.quant_flags_accepted)]
        except KeyError as e:
            raise RuntimeError(f"Missing required column '{cnst.QUANT_FLAG_DF_KEY}' in input DataFrame.") from e
            
        # Get fitted element lines for elements in the standard
        all_fitted_el_lines = [
            el_line for el_line in self._th_peak_energies.keys()
            if el_line in data_filtered_df.columns
        ]
        fitted_std_el_lines = [
            el_line for el_line in all_fitted_el_lines
            if el_line.split("_")[0] in self.detectable_els_sample
        ]

        # Update lists of measured PB ratios, their means, stddev, and corresponding theoretical energies
        std_ref_lines = {}
        for el_line in fitted_std_el_lines:
            meas_PB_ratios = data_filtered_df[el_line].tolist()
            if len(meas_PB_ratios) > 0:
                std_ref_lines[el_line] = {
                    cnst.PB_RATIO_KEY: meas_PB_ratios,
                    cnst.MEAN_PB_KEY: float(np.nanmean(meas_PB_ratios)),
                    cnst.STDEV_PB_DF_KEY: float(np.nanstd(meas_PB_ratios)),
                    cnst.PEAK_TH_ENERGY_KEY: self._th_peak_energies[el_line]
                }

        return std_ref_lines
        
        
    def _calc_corrected_PB(
        self,
        std_ref_lines: Dict[str, Dict[str, Union[float, List[float]]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ZAF-corrected Peak-to-Background (PB) ratios for the experimental standard.
    
        This method applies ZAF (atomic number, bnackscattering, absorption) corrections
        to the measured PB ratios of the standard's element lines. The corrected PB ratios are normalized
        by the mass fraction of each element to obtain the pure element PB ratios.
    
        Parameters
        ----------
        std_ref_lines : Dict[str, Dict[str, float | List[float]]]
            Dictionary mapping each element line (e.g., 'Fe_Ka') to its PB ratio statistics and theoretical peak energy.
            Must contain:
            - cnst.PEAK_TH_ENERGY_KEY: float, theoretical peak energy.
            - cnst.MEAN_PB_KEY: float, mean measured PB ratio.
            - Corresponding element's mass fraction in `self.exp_stds_cfg.w_frs`.
    
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - PB_corrected (np.ndarray): ZAF-corrected PB ratios, normalized by mass fractions.
            - Z_sample (np.ndarray): ZAF correction factors for the sample.
    
        Notes
        -----
        - Expects std_ref_lines to be non-empty
        - Relies on `self.exp_stds_cfg.w_frs` for element mass fractions.
        - Uses `Quant_Corrections.get_ZAF_mult_f_pb` for ZAF factor computation.
        - Assumes `self.detectable_els_sample` contains the list of detectable element symbols.
        """
        peak_energies_dict: Dict[str, float] = {}
        means_PB: List[float] = []
        w_frs: List[float] = []
    
        # Extract peak energies, mean PB ratios, and corresponding mass fractions
        for el_line, el_line_dict in std_ref_lines.items():
            try:
                peak_energies_dict[el_line] = el_line_dict[cnst.PEAK_TH_ENERGY_KEY]
                means_PB.append(float(el_line_dict[cnst.MEAN_PB_KEY]))
            except KeyError as e:
                raise RuntimeError(f"Missing expected key in std_ref_lines for element line '{el_line}'.") from e
    
            el = el_line.split('_')[0]
            try:
                w_frs.append(self.exp_stds_cfg.w_frs[el])
            except KeyError as e:
                raise RuntimeError(f"Mass fraction for element '{el}' not found in exp_stds_cfg.w_frs.") from e
    
        # Initialize ZAF correction calculator (second-order corrections for PB method)
        ZAF_calculator = Quant_Corrections(
            elements=self.detectable_els_sample,
            beam_energy=self.measurement_cfg.beam_energy_keV,
            emergence_angle=self.measurement_cfg.emergence_angle,
            meas_mode=self.measurement_cfg.mode,
            verbose=False
        )
    
        # Get nominal mass fractions for all detectable elements
        missing_elements = [el for el in self.detectable_els_sample if el not in self.exp_stds_cfg.w_frs]
        if missing_elements:
            raise RuntimeError(
                f"Missing mass fraction(s) for detectable element(s): {', '.join(missing_elements)}."
            )
        
        nominal_w_frs = [self.exp_stds_cfg.w_frs[el] for el in self.detectable_els_sample]

        # Calculate ZAF corrections
        # Returns arrays with dimensions corresponding to w_frs
        ZAF_pb, Z_sample = ZAF_calculator.get_ZAF_mult_f_pb(nominal_w_frs, peak_energies_dict)
    
        # Apply ZAF correction and normalize by mass fractions to get pure element PB ratios
        PB_corrected = ZAF_pb * np.array(means_PB) / np.array(w_frs)
    
        return PB_corrected, Z_sample
    
    
    def _save_std_results(
        self,
        std_ref_lines: Dict[str, Dict[str, Any]],
        PB_corrected: List[float]
    ) -> Optional[pd.DataFrame]:
        """
        Save and return a summary table of standard reference line results.
    
        Constructs a table summarizing the mean, standard deviation, relative error,
        and number of spectra used for each reference line, then saves it as a CSV file.
    
        Parameters
        ----------
        std_ref_lines : Dict
            Dictionary mapping line identifiers (str) to dictionaries containing
            statistical results for each line. Each inner dictionary must contain keys
            for mean, standard deviation, and a list or Series of PB ratios.
        PB_corrected : list of float
            List of corrected PB values, one per reference line (order must match keys of std_ref_lines).
    
        Returns
        -------
        results_df : pandas.DataFrame or None
            DataFrame with summary statistics for each reference line, or None if no lines are provided.
    
        Raises
        ------
        ValueError
            If the number of PB_corrected values does not match the number of reference lines.
    
        Notes
        -----
        The DataFrame is saved as a CSV file in Std_measurements, with a filename
        including the measurement mode and output filename suffix.
        """
        if not std_ref_lines:
            return None
    
        # Extract statistics for each reference line
        means_PB = []
        stdevs_PB = []
        n_spectra_per_line = []
        line_keys = list(std_ref_lines.keys())
    
        if len(PB_corrected) != len(line_keys):
            raise ValueError("Length of PB_corrected does not match number of reference lines.")
    
        for el_line in line_keys:
            el_line_dict = std_ref_lines[el_line]
            means_PB.append(el_line_dict[cnst.MEAN_PB_KEY])
            stdevs_PB.append(el_line_dict[cnst.STDEV_PB_DF_KEY])
            pb_ratios = el_line_dict[cnst.PB_RATIO_KEY]
            n_spectra_used = sum(
                (x is not None) and (not (isinstance(x, float) and np.isnan(x)))
                for x in pb_ratios
            )
            n_spectra_per_line.append(n_spectra_used)
    
        # Construct the results DataFrame
        results_df = pd.DataFrame({
            cnst.MEAS_PB_DF_KEY: means_PB,
            cnst.STDEV_PB_DF_KEY: stdevs_PB,
            cnst.COR_PB_DF_KEY: PB_corrected,
            cnst.REL_ER_PERCENT_PB_DF_KEY: np.array(stdevs_PB) / np.array(means_PB) * 100,
            cnst.N_SP_USED_KEY: n_spectra_per_line
        }, index=line_keys)
    
        # Save the DataFrame as CSV
        filename = f"{cnst.STDS_RESULT_FILENAME}_{self.measurement_cfg.mode}" + self.output_filename_suffix
        results_path = os.path.join(self.sample_result_dir, filename + '.csv')
        results_df.to_csv(results_path, index=True, header=True)
    
        return results_df
    
    
    def _load_xsp_standards(self) -> Tuple[dict, str]:
        """
        Load the X-ray Spectroscopy standards library for the current measurement configuration.
        
        This function attempts to load an existing standards library based on the
        measurement type and beam energy defined in the measurement configuration.
        If the library cannot be found, a new empty dictionary is created for the
        current measurement mode. If loading fails due to an unexpected error, a
        RuntimeError is raised with the original exception preserved.
        
        The function also handles copying of the reference standards to the project
        folder during reference standard measurements when exp_stds_cfg.generate_separate_std_dict = True.
        
        Returns:
            tuple[dict, str]: 
                A tuple containing:
                - standards (dict): The standards library, indexed by measurement mode.
                - stds_filepath (str): The path to the reference standards .json file.
        
        Raises:
            RuntimeError: If loading the standards library fails due to an 
                          unexpected error (not just missing files).
        """
        meas_mode = self.measurement_cfg.mode
        update_separate_std_dict = self.exp_stds_cfg.is_exp_std_measurement and self.exp_stds_cfg.generate_separate_std_dict
        
        # Load or create standards dictionary
        if self.standards_dict is None:
            
            # Determine directory of standards dict
            std_f_dir = None # Loads default std_dict
            if update_separate_std_dict or self.quant_cfg.use_project_specific_std_dict:
                # Load and save std_dict to project directory, assumed to be up 1 level from the sample directory
                project_dir = os.path.dirname(self.sample_result_dir)
                std_f_dir = project_dir
                
            try:
                standards, stds_filepath = calibs.load_standards(self.measurement_cfg.type, self.measurement_cfg.beam_energy_keV, std_f_dir = std_f_dir)
            except FileNotFoundError:
                stds_filepath = calibs.standards_dir
                standards = {meas_mode: {}}
            except Exception as e:
                raise RuntimeError("Failed to load standards library.") from e
            else:
                # Check if it needs to copy the reference standards files to the project folder
                if update_separate_std_dict and os.path.dirname(stds_filepath) != project_dir:
                    stds_filepath = shutil.copy(stds_filepath, project_dir) # Copy standards to project folder
        else:
            standards = self.standards_dict
            stds_filepath = ''

        # Ensure measurement mode exists in the standards dictionary, otherwise create it
        if meas_mode not in standards:
            standards[meas_mode] = {}
            
        return standards, stds_filepath
    
    
    def _update_standard_library(
        self,
        std_ref_lines: Dict[str, Dict[str, Union[float, List[float]]]],
        results_df: pd.DataFrame,
        Z_sample: np.ndarray
    ) -> None:
        """
        Update the standards library with new Peak-to-Background (PB) ratio measurements.
    
        This method:
        1. Loads the current standards library from disk (or creates a new one if missing).
        2. Removes any previous entries for the current standard.
        3. Appends the new measurements for each element line.
        4. Recalculates the 'Mean' reference PB ratio and associated uncertainty for each element line.
        5. Saves the updated standards library back to disk.
    
        Parameters
        ----------
        std_ref_lines : Dict[str, Dict[str, float | List[float]]]
            Dictionary mapping element lines (e.g., 'Fe_Ka') to PB ratio data and metadata.
        results_df : pd.DataFrame
            DataFrame containing measured PB ratios, corrected PB ratios, standard deviations,
            and relative errors for each element line.
        Z_sample : np.ndarray
            Mean sampel atomic number, computed using different methods.
    
        Raises
        ------
        RuntimeError
            If the standards library cannot be loaded or updated due to missing keys or invalid data.
        """
        meas_mode = self.measurement_cfg.mode
        
        # Load standards
        if self.standards_dict is not None:
            warnings.warn("The 'standards_dict' provided when initializing EMXSp_Composition_Analyzer will be ignored."
                          f"Loading standards dictionary from XSp_calibs/{self.microscope_cfg.ID}", UserWarning())
            self.standards_dict = None
        standards, stds_filepath = self._load_xsp_standards()
        
        std_lib = standards[meas_mode]
        
        # Remove all previous entries measured from this standard
        was_standard_already_measured = False
        for el_line, stds_list in list(std_lib.items()):
            for i, std_dict in enumerate(list(stds_list)):
                if std_dict.get(cnst.STD_ID_KEY) == self.sample_cfg.ID:
                    std_lib[el_line].pop(i)
                    was_standard_already_measured = True
                    break
        if was_standard_already_measured and self.verbose:
            print_single_separator()
            print(f"Previously measured values for standard '{self.sample_cfg.ID}' were found and removed.")
    
        # Add new standards
        now = datetime.now()
        for el_line in std_ref_lines.keys():
            # Validate presence of required result_df fields
            for key in [
                cnst.COR_PB_DF_KEY,
                cnst.MEAS_PB_DF_KEY,
                cnst.STDEV_PB_DF_KEY,
                cnst.REL_ER_PERCENT_PB_DF_KEY
            ]:
                if key not in results_df.columns:
                    raise RuntimeError(f"Missing required column '{key}' in results_df.")
    
            std_dict_new = {
                cnst.STD_ID_KEY: self.sample_cfg.ID,
                cnst.STD_FORMULA_KEY: self.exp_stds_cfg.formula,
                cnst.STD_TYPE_KEY: self.sample_cfg.type,
                cnst.DATETIME_KEY: now.strftime("%Y-%m-%d %H:%M:%S"),
                cnst.COR_PB_DF_KEY: results_df.at[el_line, cnst.COR_PB_DF_KEY],
                cnst.MEAS_PB_DF_KEY: results_df.at[el_line, cnst.MEAS_PB_DF_KEY],
                cnst.STDEV_PB_DF_KEY: results_df.at[el_line, cnst.STDEV_PB_DF_KEY],
                cnst.REL_ER_PERCENT_PB_DF_KEY: results_df.at[el_line, cnst.REL_ER_PERCENT_PB_DF_KEY],
                cnst.STD_USE_FOR_MEAN_KEY : self.exp_stds_cfg.use_for_mean_PB_calc,
                cnst.STD_Z_KEY: Z_sample
            }
    
            # Add or append standard measurement
            if el_line in std_lib:
                if self.verbose:
                    print(f"Added the measured standard PB value for {el_line} to the current list.")
                std_lib[el_line].append(std_dict_new)
            else:
                if self.verbose:
                    print(f"Created a new list for the {el_line} line PB standard values.")
                std_lib[el_line] = [std_dict_new]
    
            # Recalculate mean of standards (excluding previous mean entry)
            std_el_line_entries = [
                std for std in std_lib[el_line]
                if std.get(cnst.STD_ID_KEY) != cnst.STD_MEAN_ID_KEY
            ]
            # Select corrected PB ratios that should be used for calculating the mean (i.e., PB ratios computed from the mean)
            list_PB_for_mean = [std[cnst.COR_PB_DF_KEY] for std in std_el_line_entries if std[cnst.STD_USE_FOR_MEAN_KEY]]
            if len(list_PB_for_mean) > 0: 
                mean_PB = float(np.mean(list_PB_for_mean)) if list_PB_for_mean else float("nan")
                stddev_mean_PB = float(np.std(list_PB_for_mean)) if list_PB_for_mean else float("nan")
                error_mean_PB = (stddev_mean_PB / mean_PB * 100) if mean_PB else float("nan")
        
                std_dict_mean = {
                    cnst.STD_ID_KEY: cnst.STD_MEAN_ID_KEY,
                    cnst.DATETIME_KEY: now.strftime("%Y-%m-%d %H:%M:%S"),
                    cnst.COR_PB_DF_KEY: mean_PB,
                    cnst.STDEV_PB_DF_KEY: stddev_mean_PB,
                    cnst.REL_ER_PERCENT_PB_DF_KEY: error_mean_PB
                }
                std_el_line_entries.append(std_dict_mean)
            std_lib[el_line] = std_el_line_entries
    
        # Save updated file with standards
        try:
            with open(stds_filepath, "w") as file:
                json.dump(standards, file, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save updated standards to {stds_filepath}.") from e


# Delegation to extracted composition_analysis modules.
# This preserves the current public class/API while routing implementation
# to the new modular structure.
EMXSp_Composition_Analyzer._find_optimal_k = ClusteringModule._find_optimal_k
EMXSp_Composition_Analyzer._get_most_freq_k = ClusteringModule._get_most_freq_k
EMXSp_Composition_Analyzer._get_k = ClusteringModule._get_k
EMXSp_Composition_Analyzer._is_single_cluster = ClusteringModule._is_single_cluster
EMXSp_Composition_Analyzer._run_kmeans_clustering = ClusteringModule._run_kmeans_clustering
EMXSp_Composition_Analyzer._prepare_composition_dataframes = ClusteringModule._prepare_composition_dataframes
EMXSp_Composition_Analyzer._get_clustering_kmeans = ClusteringModule._get_clustering_kmeans
EMXSp_Composition_Analyzer._get_clustering_dbscan = ClusteringModule._get_clustering_dbscan
EMXSp_Composition_Analyzer._compute_cluster_statistics = ClusteringModule._compute_cluster_statistics

EMXSp_Composition_Analyzer._correlate_centroids_to_refs = ReferenceMatchingModule._correlate_centroids_to_refs
EMXSp_Composition_Analyzer._assign_reference_phases = ReferenceMatchingModule._assign_reference_phases
EMXSp_Composition_Analyzer._get_ref_confidences = ReferenceMatchingModule._get_ref_confidences

EMXSp_Composition_Analyzer._save_plots = PlottingModule._save_plots
EMXSp_Composition_Analyzer._load_custom_plot_function = PlottingModule._load_custom_plot_function
EMXSp_Composition_Analyzer._run_custom_clustering_plot = PlottingModule._run_custom_clustering_plot
EMXSp_Composition_Analyzer._save_clustering_plot = PlottingModule._save_clustering_plot
EMXSp_Composition_Analyzer._save_violin_plot_powder_mixture = PlottingModule._save_violin_plot_powder_mixture
EMXSp_Composition_Analyzer._save_silhouette_plot = PlottingModule._save_silhouette_plot

EMXSp_Composition_Analyzer._compile_standards_from_references = StandardsModule._compile_standards_from_references
EMXSp_Composition_Analyzer._fit_stds_and_save_results = StandardsModule._fit_stds_and_save_results
EMXSp_Composition_Analyzer._evaluate_exp_std_fit = StandardsModule._evaluate_exp_std_fit
EMXSp_Composition_Analyzer._assemble_std_PB_data = StandardsModule._assemble_std_PB_data
EMXSp_Composition_Analyzer._calc_corrected_PB = StandardsModule._calc_corrected_PB
EMXSp_Composition_Analyzer._save_std_results = StandardsModule._save_std_results
EMXSp_Composition_Analyzer._load_xsp_standards = StandardsModule._load_xsp_standards
EMXSp_Composition_Analyzer._update_standard_library = StandardsModule._update_standard_library
 