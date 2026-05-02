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
from pymatgen.core.composition import Composition
from sklearn.cluster import KMeans
import cvxpy as cp
 

# Project-specific imports
from autoemx.core.quantifier import XSp_Quantifier
from autoemx.core.em_runtime.controller import EM_Controller
from autoemx.core.em_runtime.sample_finder import EM_Sample_Finder
import autoemx.calibrations as calibs
import autoemx.utils.constants as cnst
import autoemx.config.defaults as dflt
from autoemx.utils import (
    print_single_separator,
    print_double_separator,
    to_latex_formula,
    make_unique_path,
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
from autoemx.config.ledger_schemas import (

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
from autoemx.config.schema_models import EDSStandardsFile
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

from autoemx._logging import get_logger
logger = get_logger(__name__)

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
    initial_clustering_cfg : autoemx.config.ledger_schemas.ClusteringConfig
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
    standards_dict : autoemx.config.schema_models.EDSStandardsFile | dict, optional
        Standards payload for reference PB values from experimental standards.
        If a dict is provided, it must conform to the standards Pydantic schema (or legacy
        shape that can be normalized by the schema compatibility adapter).
        If None, standards are loaded from the calibration directory for the active microscope.
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
        standards_dict: Optional[Union[EDSStandardsFile, Dict[str, Any]]] = None,
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
            logger.info(f"▶️ Starting compositional analysis of sample {sample_cfg.ID}")
            
            
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
        if standards_dict is None:
            self.standards = None
        elif isinstance(standards_dict, EDSStandardsFile):
            self.standards = standards_dict
        elif isinstance(standards_dict, dict):
            self.standards = EDSStandardsFile.from_payload(
                standards_dict,
                meas_type=self.measurement_cfg.type,
                beam_energy_keV=int(self.measurement_cfg.beam_energy_keV),
            )
        else:
            raise TypeError(
                "standards_dict must be None, EDSStandardsFile, or a dictionary payload"
            )

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
                # When a project directory is provided, save under a per-sample folder.
                # If the provided path already points to a sample folder (resume), reuse it.
                normalized_results_dir = os.path.normpath(results_dir)
                is_sample_dir = os.path.basename(normalized_results_dir) == sample_cfg.ID
                has_sample_artifacts = (
                    os.path.exists(os.path.join(results_dir, cnst.LEDGER_FILENAME + cnst.LEDGER_FILEEXT))
                    or os.path.exists(os.path.join(results_dir, cnst.IMAGES_DIR))
                    or os.path.exists(os.path.join(results_dir, cnst.SPECTRA_DIR))
                )

                if is_sample_dir or has_sample_artifacts:
                    sample_result_dir = results_dir
                else:
                    sample_result_dir = os.path.join(results_dir, sample_cfg.ID)

                os.makedirs(sample_result_dir, exist_ok=True)
                results_dir = sample_result_dir

        self.sample_result_dir = results_dir
        self.output_filename_suffix = output_filename_suffix
        self.verbose = verbose

        if is_XSp_measurement:
            # --- Variable initialization
            self.standards_dict = None
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
        spectra_dir = self._get_spectra_dir()
        os.makedirs(spectra_dir, exist_ok=True)
        ledger_path = self._get_ledger_path()
        if os.path.exists(ledger_path):
            return
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
        base_name = f"analysis_quant{quantification_id}_clust{clustering_id}"
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
        lines.append("QUANTIFICATION CONFIGURATIONS")
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
            reference_lines_by_element = [
                el_line
                for el, el_line in qc.reference_lines_by_element.items()
                if el in sample_elements
            ]
            ref_lines_str = ', '.join(reference_lines_by_element)
            if ref_lines_str:
                lines.append(f"  Reference lines      : {ref_lines_str}")

        # Clustering section
        cc = self.clustering_cfg

        lines.append("")
        lines.append("=" * 60)
        lines.append("CLUSTERING CONFIGURATIONS")
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
    
        This method determines how the `standards_dict` attribute is initialised
        based on the sample configuration and measurement type:
    
        - If the measurement is of a known powder mixture, the standards dictionary
          is compiled from reference data using `_compile_standards_from_references()`.
    
        - Otherwise, the standards dictionary is expected to be loaded directly
          within the `XSp_Quantifier` and is set to `None` here.
    
        Returns
        -------
        None
            This method modifies the `self.standards_dict` attribute in place.
        """
        is_known_mixture = getattr(self.powder_meas_cfg, "is_known_powder_mixture_meas", False)
        
        if is_known_mixture:
            self.standards_dict = self._compile_standards_from_references()
        elif self.quant_cfg.use_project_specific_std_dict:
            standards, _ = self._load_standards()
            self.standards_dict = standards.to_standards_dict().get(self.measurement_cfg.mode, {})
        else:
            # Standards dictionary will be loaded directly within the `XSp_Quantifier`
            self.standards_dict = None

    def _compile_standards_from_references(self) -> dict:
        """Forward to StandardsModule implementation."""
        return StandardsModule._compile_standards_from_references(self)

    def _load_standards(self) -> Tuple[EDSStandardsFile, str]:
        """Forward to StandardsModule implementation."""
        return StandardsModule._load_standards(self)

    def _load_xsp_standards(self):
        return StandardsModule._load_xsp_standards(self)

    def _fit_stds_and_save_results(self):
        return StandardsModule._fit_stds_and_save_results(self)

    def _evaluate_exp_std_fit(self, tot_n_spectra: int) -> Tuple[bool, bool]:
        return StandardsModule._evaluate_exp_std_fit(self, tot_n_spectra)

    def _assemble_std_PB_data(self, data_df: 'pd.DataFrame'):
        return StandardsModule._assemble_std_PB_data(self, data_df)

    def _calc_corrected_PB(self, std_ref_lines):
        return StandardsModule._calc_corrected_PB(self, std_ref_lines)

    def _save_std_results(self, std_ref_lines, pb_corrected, z_sample):
        return StandardsModule._save_std_results(self, std_ref_lines, pb_corrected, z_sample)

    @staticmethod
    def _serialize_standard_mean_z(z_sample: Any):
        return StandardsModule._serialize_standard_mean_z(z_sample)

    def _update_standard_library(self, fit_results) -> None:
        return StandardsModule._update_standard_library(self, fit_results)


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
        msa_file_path: str,
    ) -> int:
        """
        Acquire an X-ray spectrum at the specified stage position and store the results.

        Parameters
        ----------
        x, y : float
            The x, y machine coordinates for the spectrum acquisition.
        spectrum_id : str
            Unique identifier for the spectrum.
        msa_file_path : str
            Path to save the acquired spectrum (.msa file).

        Returns
        -------
        total_counts : int
            Total counts in the acquired spectrum, derived from persisted data.

        Notes
        -----
        All spectrum data is saved to and loaded from .msa files. No in-memory spectrum caching is used.
        """
        # Acquire at the instrument and rely on persisted pointer files as source of truth.
        background_elements = None
        if self.quant_cfg.use_instrument_background:
            background_elements = list(getattr(self, "detectable_els_sample", []) or [])

        os.makedirs(os.path.dirname(msa_file_path), exist_ok=True)

        # Measure real acquisition time
        _acq_start = time.time()
        spectrum_data, background_data = self.EM_controller.acquire_XS_spot_spectrum(
            x, y,
            self.measurement_cfg.max_acquisition_time,
            self.measurement_cfg.target_acquisition_counts,
            elements=background_elements,
            msa_file_path=msa_file_path,
        )
        _acq_end = time.time()
        measured_real_time = _acq_end - _acq_start

        counts_arr = np.asarray(spectrum_data, dtype=float)

        pointer_path = Path(msa_file_path)
        if not pointer_path.exists():
            from autoemx.utils.legacy.spectrum_pointer_writer import write_spectrum_pointer_file
            write_spectrum_pointer_file(
                str(pointer_path),
                list(counts_arr),
                getattr(self, 'energy_vals', []),
                real_time=measured_real_time
            )
        loaded_counts = SampleLedger._load_counts_from_pointer_file(pointer_path)
        counts_arr = np.asarray(loaded_counts, dtype=float)

        # For user info, print the acquisition time
        if self.verbose:
            logger.info(f"✅ Acquisition took {measured_real_time:.2f} s")

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

        return int(np.round(np.sum(counts_arr)))
    
    
    def _fit_exp_std_spectrum(
        self,
        spectrum: Iterable,
        background: Optional[Iterable] = None,
        sp_collection_time: float = None,
        els_w_frs: Optional[Dict[str,float]] = None,
        sp_id: str = '',
        verbose: bool = True
    ) -> QuantificationResult:
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
        QuantificationResult
            Schema result for this spectrum including quant flag/comment diagnostics and
            fit metrics under ``fit_result``.
    
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
            logger.info('🔬 Fitting spectrum' + sp_id_str)
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
            return quant_record
        
        try:
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
                standards_dict=self.standards_dict,
                verbose=False,
                fitting_verbose=False
            )
            bad_quant_flag = quantifier.initialize_and_fit_spectrum(print_results=self.verbose)
            is_fit_valid = True
            min_bckgrnd_ref_lines = quantifier._get_min_bckgrnd_cnts_ref_quant_lines()
        except Exception as e:
            is_fit_valid = False
            logger.error(f"❌ {type(e).__name__}: {e}")
            traceback.print_exc()
            quant_flag, comment = self._check_fit_quant_validity(is_fit_valid, None, None, None)
            quant_record = QuantificationResult(
                quantification_id=self.current_quantification_id,
                quant_flag=quant_flag,
                comment=comment,
                diagnostics=QuantificationDiagnostics(
                    converged=False,
                    interrupted=True,
                ),
            )
            return quant_record
        
        _, are_all_ref_peaks_present = self._assemble_fit_info(quantifier)
        
        if are_all_ref_peaks_present:
            quant_flag, comment = self._check_fit_quant_validity(is_fit_valid, bad_quant_flag, quantifier, min_bckgrnd_ref_lines)
        else:
            comment = "Reference peak missing"
            quant_flag = 10
        
        if verbose:
            fit_time = time.time() - start_quant_time
            logger.info(f"✅ Fitting took {fit_time:.2f} s")

        quant_record = QuantificationResult(
            quantification_id=self.current_quantification_id,
            quant_flag=quant_flag,
            comment=comment,
            fit_result=quantifier.export_fit_result(),
            diagnostics=QuantificationDiagnostics(
                interrupted=False,
                min_background_ref_lines=min_bckgrnd_ref_lines,
            ),
        )

        return quant_record
    
    
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
                            logger.warning(f"⚠️ {el_line} reference peak missing.")
                    
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
    ) -> Tuple[Optional[Dict[str, Any]], QuantificationResult, Optional[float]]:
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
        Tuple[Optional[Dict[str, Any]], QuantificationResult, Optional[float]]
            Runtime quantifier payload (or None), persisted schema result, and elapsed quantification time.
    
        Notes
        -----
        - Filtering flags are appended through function _check_fit_quant_validity().
        """
        quantification_time = None
        if verbose:
            if sp_id != '':
                sp_id_str = " #" + sp_id
            else:
                sp_id_str = '...'
            print_single_separator()
            logger.info('🔬 Quantifying spectrum' + sp_id_str)
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
            if verbose:
                quantification_time = time.time() - start_quant_time
            return None, quant_record, quantification_time

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
            standards_dict=self.standards_dict,
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
            logger.error(f"❌ {type(e).__name__}: {e}")
            is_quant_fit_valid = False
            quant_flag, comment = self._check_fit_quant_validity(is_quant_fit_valid, None, None, None)
            quant_record = quantifier.export_quantification_result(
                quantification_id=self.current_quantification_id,
                quant_result=None,
                quant_flag=quant_flag,
                comment=comment,
            )
            if verbose:
                quantification_time = time.time() - start_quant_time
            return None, quant_record, quantification_time
        else:
            quant_flag, comment = self._check_fit_quant_validity(is_quant_fit_valid, bad_quant_flag, quantifier, min_bckgrnd_ref_lines)
            quant_record = quantifier.export_quantification_result(
                quantification_id=self.current_quantification_id,
                quant_result=quant_result,
                quant_flag=quant_flag,
                comment=comment,
            )

        if verbose:
            quantification_time = time.time() - start_quant_time

        return quant_result, quant_record, quantification_time


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
                counts_arr = np.asarray(counts, dtype=float)
            except Exception:
                continue

            # Only hydrate entries with valid counts (non-empty, non-NaN, positive sum)
            if counts_arr.size == 0 or not np.isfinite(np.sum(counts_arr)) or np.sum(counts_arr) <= 0:
                continue

            spectra_vals.append(counts_arr)

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
            if self.standards_dict is not None and not load_if_missing:
                standards_by_line = self.standards_dict
            else:
                standards, _ = self._load_standards()
                standards_by_line = standards.standards_by_mode.get(self.measurement_cfg.mode, {})
        else:
            standards_by_line = self.standards_dict
            if standards_by_line is None and self.standards is not None:
                if isinstance(self.standards, EDSStandardsFile):
                    standards_by_line = self.standards.standards_by_mode.get(
                        self.measurement_cfg.mode,
                        {},
                    )
                elif self.measurement_cfg.mode in self.standards:
                    standards_by_line = self.standards[self.measurement_cfg.mode]
                else:
                    standards_by_line = self.standards
            if standards_by_line is None and load_if_missing:
                standards, _ = self._load_standards()
                standards_by_line = standards.standards_by_mode.get(self.measurement_cfg.mode, {})

        if standards_by_line is None:
            return {}

        relevant_elements = set(self.detectable_els_sample) | set(self.detectable_els_substrate)
        reference_values_by_el_line: Dict[str, Any] = {}
        if self.quant_cfg.method == "PB":
            for el_line, std_values in standards_by_line.items():
                element = el_line.split("_", maxsplit=1)[0]
                if element not in relevant_elements:
                    continue

                if hasattr(std_values, "reference_mean"):
                    reference_mean = getattr(std_values, "reference_mean", None)
                    if reference_mean is None:
                        continue
                    reference_values_by_el_line[el_line] = float(reference_mean.corrected_pb)
                    continue

                if isinstance(std_values, dict):
                    reference_mean = std_values.get("reference_mean")
                    if isinstance(reference_mean, dict) and cnst.COR_PB_DF_KEY in reference_mean:
                        reference_values_by_el_line[el_line] = float(reference_mean[cnst.COR_PB_DF_KEY])
                        continue
                    std_values = std_values.get("entries", [])

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
            return 0
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
        label = f"Quant {quantification_id} {self.output_filename_suffix}"
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
            els_w_fr=self._get_forced_mass_fractions(),
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
            els_w_fr=self._get_forced_mass_fractions(),
            options=self._build_quantification_options(),
            reference_values_by_el_line=reference_values_by_el_line,
            reference_lines_by_element=reference_lines_by_element,
            clustering_configs=[self._build_clustering_config_descriptor(clustering_id=0)],
            active_clustering_cfg_index=0,
        )


    def _get_forced_mass_fractions(self) -> Optional[Dict[str, float]]:
        """Return forced elemental mass fractions for the active run, if any."""
        if getattr(self.exp_stds_cfg, "is_exp_std_measurement", False):
            forced = getattr(self.exp_stds_cfg, "w_frs", None)
            if isinstance(forced, dict) and forced:
                return {str(el): float(fr) for el, fr in forced.items()}

        forced = getattr(self.sample_cfg, "w_frs", None)
        if isinstance(forced, dict) and forced:
            return {str(el): float(fr) for el, fr in forced.items()}

        return None


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
                logger.warning("⚠️ Flagged for poor fit")
            comment = start_str_comments + "poor fit"
            quant_flag = 4
        elif bad_quant_flag == 2:
            if self.verbose and is_quant_fit_valid:
                logger.warning("⚠️ Flagged for excessively high analytical error")
            comment = start_str_comments + "excessively high analytical error"
            quant_flag = 5
        elif bad_quant_flag == 3:
            if self.verbose and is_quant_fit_valid:
                logger.warning("⚠️ Flagged for excessive X-ray absorption")
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
                logger.error("❌ Error during spectrum collection. No quantification was done.")
        elif np.sum(spectrum) < 0.9 * self.measurement_cfg.target_acquisition_counts:
            # Skip quantification of spectrum when counts are too low
            is_spectrum_valid = False
            comment = "Total counts too low"
            quant_flag = 2
            if self.verbose:
                logger.info(f"⏭️ Quantification skipped due to spectrum counts lower than 90% of the target counts of {self.measurement_cfg.target_acquisition_counts}")
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
                    logger.info(f"⏭️ Quantification skipped due to at least {n_vals_considered} spectrum points with E < {en_threshold} keV having a count lower than {min_background_threshold}")
                    logger.warning("⚠️ This generally indicates an excessive absorption of X-rays before they reach the detector, which compromises accurate measurements of PB ratios.")
    
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
                    logger.warning(f"⚠️ Intensity of substrate element {el} is {peak_int:.0f} cnts, larger than {sub_peak_int_threshold}% of total counts")
                    logger.warning("⚠️ This is likely to lead to large quantification errors.")
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
                    logger.warning(f"⚠️ Counts below a reference peak are on average < {min_background_threshold}")
                    logger.warning("⚠️ This is likely to lead to large quantification errors.")
            else:
                quant_flag = 0  # Quantification is ok
    
        return is_spectrum_valid, quant_flag, comment  # Not used, but returned for completeness
    
    
    #%% Spectra acquisition and quantification routines
    # ============================================================================= 
    def _collect_spectra(
        self,
        n_spectra_to_collect: int,
        n_tot_sp_collected: Optional[int] = None,
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

        # Auto-detect next available spectrum ID if not provided
        if n_tot_sp_collected is None:
            pointer_files = self._list_pointer_files_in_spectra_dir()
            max_id = -1
            for pf in pointer_files:
                stem = pf.stem
                if stem.startswith(cnst.SPECTRUM_FILENAME_PREFIX):
                    spectrum_id = stem[len(cnst.SPECTRUM_FILENAME_PREFIX):]
                    if spectrum_id.isdigit():
                        max_id = max(max_id, int(spectrum_id))
            n_tot_sp_collected = max_id + 1

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
                    logger.info(f'🔬 Acquiring spectrum #{n_tot_sp_collected}...')

                current_spectrum_id = str(n_tot_sp_collected)
                spectrum_relpath = self._build_spectrum_relpath(current_spectrum_id)
                manufacturer_msa_path = os.path.join(self.sample_result_dir, spectrum_relpath)
                os.makedirs(os.path.dirname(manufacturer_msa_path), exist_ok=True)

                n_tot_sp_collected += 1
                total_counts = self._acquire_spectrum(
                    x,
                    y,
                    spectrum_id=current_spectrum_id,
                    msa_file_path=manufacturer_msa_path,
                )

                # Immediately update ledger after each successful acquisition
                try:
                    from autoemx.config.ledger_schemas import SpectrumEntry, SampleLedger
                except ImportError:
                    pass  # Already imported at top
                # Build SpectrumEntry for this spectrum
                spectrum_index = n_tot_sp_collected - 1
                spectrum_entry = self._build_spectrum_entry(spectrum_index)
                # Load or create ledger
                ledger_path = self._get_ledger_path()
                try:
                    ledger = self._load_existing_ledger()
                except Exception:
                    ledger = None
                if ledger is None:
                    ledger = SampleLedger(
                        sample_id=self.sample_cfg.ID,
                        sample_path=os.path.abspath(self.sample_result_dir),
                        configs=self._build_ledger_configs(),
                        spectra=[],
                        quantification_configs=[],
                        active_quant=None,
                    )
                ledger.spectra.append(spectrum_entry)
                ledger.to_json_file(ledger_path)

                # Contamination check: skip quantification if counts are too low (only at first measurement spot)
                if i==0 and self.sample_cfg.is_particle_acquisition:
                    if total_counts < 0.95 * self.measurement_cfg.target_acquisition_counts:
                        if quantify:
                            self.spectra_quant_records.append(None)
                        if self.verbose:
                            logger.warning('⚠️ Current particle is unlikely to be part of the sample.\nSkipping to the next particle.')
                            logger.info('ℹ️ Increase measurement_cfg.max_acquisition_time if this behavior is undesired.')
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
                        cnst.ANNOTATION_TEXT_KEY: (
                            str(n_tot_sp_collected - 1 - latest_spot_id + i),
                            (xy_center[0] - 30, xy_center[1] - 15)
                        ),
                        cnst.ANNOTATION_CIRCLE_KEY: (10, xy_center, -1)
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
                Number of CPU cores for parallel fitting.
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
                            not interrupt_fits_bad_spectra and getattr(self.spectra_quant_records[i].diagnostics, 'interrupted', False)
                        )
                    ]
                    # Always skip spectra that have been tried (quant_record exists), regardless of interrupted, unless interrupt_fits_bad_spectra is False
                    if interrupt_fits_bad_spectra:
                        indices_to_process = [
                            i for i in indices_to_process
                            if self.spectra_quant_records[i] is None
                        ]
            else:
                self._load_or_create_ledger()
                self._ensure_current_quantification_run(force_new=force_requantification)
                self._persist_current_quantification_config()
                self._ensure_quant_tracking_length(tot_spectra_collected)
                if force_requantification:
                    indices_to_process = list(range(tot_spectra_collected))
                else:
                    indices_to_process = [
                        i for i in range(tot_spectra_collected)
                        if self.spectra_quant_records[i] is None
                    ]

            n_spectra_to_quant = len(indices_to_process)
        
            if self.verbose and n_spectra_to_quant > 0:
                print_single_separator()
                quant_str = "quantification" if quantify else "fitting"
                logger.info(f"▶️ Starting {quant_str} of {n_spectra_to_quant} spectra on up to {_n_cores} cores")
        
            # Worker returns (index, result, quant_record, quantification_time) tuple
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
                    result, quant_record, quantification_time = self._fit_quantify_spectrum(
                        spectrum,
                        background,
                        sp_collection_time,
                        sp_id,
                        spectrum_index=i,
                        interrupt_fits_bad_spectra=interrupt_fits_bad_spectra,
                    )
                else:
                    quant_record = self._fit_exp_std_spectrum(
                        spectrum,
                        background,
                        sp_collection_time,
                        sp_id=sp_id,
                    )
                    result = None
                    quantification_time = None
        
                return i, result, quant_record, quantification_time
        
            # Temporarily remove the analyzer to avoid pickling errors from 'loky' backend
            tmp_analyzer = None
            if hasattr(self, "EM_controller") and hasattr(self.EM_controller, "analyzer"):
                tmp_analyzer = self.EM_controller.analyzer
                del self.EM_controller.analyzer

            def _finalize_quant_result(idx, result, quant_record, quantification_time):
                quant_flag = quant_record.quant_flag
                comment = quant_record.comment
                had_prior_record = self.spectra_quant_records[idx] is not None
                was_interrupted = (
                    had_prior_record
                    and getattr(self.spectra_quant_records[idx].diagnostics, 'interrupted', False)
                )

                self.spectra_quant_records[idx] = quant_record
                self.spectral_data[cnst.COMMENTS_DF_KEY][idx] = comment
                self.spectral_data[cnst.QUANT_FLAG_DF_KEY][idx] = quant_flag

                if self.verbose:
                    print_single_separator()
                    lines = [""]
                    lines.append(f" Spectrum #{idx}/{tot_spectra_collected - 1}:")
                    if result is None:
                        if comment:
                            lines.append(f"  {comment}")
                    else:
                        if quantify:
                            for el, at_fr in result[cnst.COMP_AT_FR_KEY].items():
                                lines.append(f"  {el} at%: {at_fr * 100:.2f}%")
                            lines.append(f"  Analytical error: {result[cnst.AN_ER_KEY] * 100:.2f}%")
                        else:
                            lines.append("  Fit completed")
                        if quant_flag not in (None, 0) and comment:
                            lines.append(f"  {comment}")
                        if quantification_time is not None:
                            lines.append(f"  Quantification took {quantification_time:.2f} s")
                    lines.append("")
                    logger.info("\n".join(lines))

                if quant_record is not None:
                    overwrite = (requantify_only_unquantified_spectra and had_prior_record) or was_interrupted
                    self._persist_quantification_record(idx, quant_record, overwrite=overwrite)
            
            results_with_idx = []
            try:
                if quantify:
                    # Stream completed tasks back to the main process so progress is visible in real time.
                    try:
                        completed = Parallel(
                            n_jobs=_n_cores,
                            backend='loky',
                            return_as='generator_unordered',
                        )(
                            delayed(_process_one)(i) for i in indices_to_process
                        )
                    except TypeError:
                        # Compatibility fallback for joblib versions without return_as.
                        completed = Parallel(n_jobs=_n_cores, backend='loky')(
                            delayed(_process_one)(i) for i in indices_to_process
                        )

                    for idx, result, quant_record, quantification_time in completed:
                        _finalize_quant_result(idx, result, quant_record, quantification_time)
                else:
                    # Run in parallel for fitting-only path
                    results_with_idx = Parallel(n_jobs=_n_cores, backend='loky')(
                        delayed(_process_one)(i) for i in indices_to_process
                    )
            except Exception as e:
                logger.warning(f"⚠️ Parallel quantification failed ({type(e).__name__}: {e}), falling back to sequential execution.")
                if quantify:
                    # Sequential fallback, also stream per-spectrum logging/progress.
                    for i in indices_to_process:
                        idx, result, quant_record, quantification_time = _process_one(i)
                        _finalize_quant_result(idx, result, quant_record, quantification_time)
                else:
                    # Sequential fallback, also collect results
                    results_with_idx = [_process_one(i) for i in indices_to_process]
            finally:
                # Restore analyzer
                if tmp_analyzer is not None:
                    self.EM_controller.analyzer = tmp_analyzer
            
            if len(results_with_idx) > 0:
                # Sort results by original spectrum index to guarantee correct order
                results_with_idx.sort(key=lambda x: x[0])
                
                if not quantify:
                    for idx, result, quant_record, quantification_time in results_with_idx:
                        _finalize_quant_result(idx, result, quant_record, quantification_time)


    #%% Find number of clusters in kmeans
    # ============================================================================= 
    def _find_optimal_k(self, compositions_df, k, compute_k_only_once = False):
        return ClusteringModule._find_optimal_k(self, compositions_df, k, compute_k_only_once)

    @staticmethod
    def _get_most_freq_k(
        compositions_df: 'pd.DataFrame',
        max_k: int,
        k_finding_method: str,
        verbose: bool = False,
        show_plot: bool = False,
        results_dir: str = None
    ) -> int:
        return ClusteringModule._get_most_freq_k(
            compositions_df,
            max_k,
            k_finding_method,
            verbose=verbose,
            show_plot=show_plot,
            results_dir=results_dir,
        )

    @staticmethod
    def _get_k(
        compositions_df: 'pd.DataFrame',
        max_k: int = 6,
        method: str = 'silhouette',
        model: 'KMeans' = None,
        results_dir: str = None,
        show_plot: bool = False
    ) -> int:
        return ClusteringModule._get_k(
            compositions_df,
            max_k=max_k,
            method=method,
            model=model,
            results_dir=results_dir,
            show_plot=show_plot,
        )

    @staticmethod
    def _is_single_cluster(
        compositions_df: 'pd.DataFrame',
        verbose: bool = False
    ) -> bool:
        return ClusteringModule._is_single_cluster(compositions_df, verbose=verbose)

    #%% Clustering operations
    # ============================================================================= 
    def _run_kmeans_clustering(self, k, compositions_df):
        return ClusteringModule._run_kmeans_clustering(self, k, compositions_df)

    def _prepare_composition_dataframes(self, compositions_list_at, compositions_list_w):
        return ClusteringModule._prepare_composition_dataframes(self, compositions_list_at, compositions_list_w)

    def _get_clustering_kmeans(
        self,
        k: int,
        compositions_df: 'pd.DataFrame'
    ) -> Tuple['KMeans', 'np.ndarray']:
        return ClusteringModule._get_clustering_kmeans(self, k, compositions_df)

    def _get_clustering_dbscan(
        self,
        compositions_df: 'pd.DataFrame'
    ) -> Tuple['np.ndarray', int]:
        return ClusteringModule._get_clustering_dbscan(self, compositions_df)


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
        return ClusteringModule._compute_cluster_statistics(
            self,
            compositions_df,
            compositions_df_other_fr,
            centroids,
            labels,
        )
    
    
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

        # 1. Make analysis directory to save results
        self._make_analysis_dir()
    
        self._save_analysis_config_summary()

        # 2. Select compositions to use for clustering
        if max_analytical_error_percent is not None:
            max_analytical_error = max_analytical_error_percent / 100
        else:
            max_analytical_error = max_analytical_error_percent
        (compositions_list_at, compositions_list_w, unused_compositions_list,
         df_indices, n_datapts) = self._select_good_compositions(max_analytical_error)
        n_datapts_used = len(compositions_list_at)
    
        if n_datapts_used < 5:
            print_single_separator()
            logger.warning(f"⚠️ Only {n_datapts_used} spectra were considered 'good', but a minimum of 5 data points are required for clustering.")
            # Print additional messages with how many spectra were discarded for which reason
            self._report_n_discarded_spectra(n_datapts, max_analytical_error)
            # Save Composition.csv file anyways
            self._save_collected_data(None, None, backup_previous_data=True, include_spectral_data=False)
            return False, 0, 0  # zeroes are placeholders
    
        if self.verbose:
            print_single_separator()
            logger.info('ℹ️ Spectra selection:')
            logger.info(f"ℹ️ {n_datapts_used} data points are used, out of {n_datapts} collected spectra.")
            self._report_n_discarded_spectra(n_datapts, max_analytical_error)

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
            logger.warning('⚠️ Clustering via DBSCAN is not implemented yet')
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
        return ReferenceMatchingModule._correlate_centroids_to_refs(
            self,
            centroids,
            cluster_radii,
            ref_phases_df,
        )

    def _assign_reference_phases(self, centroids, rms_dist_cluster):
        return ReferenceMatchingModule._assign_reference_phases(self, centroids, rms_dist_cluster)

    @staticmethod
    def _get_ref_confidences(
        centroid: 'np.ndarray',
        ref_phases: 'np.ndarray',
        ref_names: List[str]
    ) -> Tuple[Optional[float], Dict]:
        return ReferenceMatchingModule._get_ref_confidences(centroid, ref_phases, ref_names)


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
                    cluster_mix_dict[f'{cnst.CS_MIX_DF_KEY}{i}'] = float(f"{mixture_dict[cnst.CONF_SCORE_KEY]:.2f}")
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
            logger.info(f"▶️ Starting collection{quant_str} of {tot_spectra_to_collect} spectra.")
        
        while tot_n_spectra < tot_spectra_to_collect:
            if self.verbose:
                print_double_separator()
                logger.info(f"🔬 Collecting{quant_str} {n_spectra_to_collect} spectra...")
    
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
                logger.info(f"✅ {tot_n_spectra}/{tot_spectra_to_collect} spectra collected and saved.")
                
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
                    logger.warning("⚠️ Acquisition interrupted.")
                    if self.sample_cfg.is_particle_acquisition:
                        logger.warning(f'⚠️ Not enough particles were found on the sample to collect all {tot_spectra_to_collect} spectra.')
                    elif self.sample_cfg.is_grid_acquisition:
                        logger.warning(f'⚠️ The specified spectrum spacing did not allow to collect all {tot_spectra_to_collect} spectra.\n'
                              "Change spacing in bulk_meas_cfg to collect more spectra.")
                break
    
        print_double_separator()
        logger.info('ℹ️ Sample ID: %s', self.sample_cfg.ID)
        par_str = f' over {self.particle_cntr} particles' if self.sample_cfg.is_particle_acquisition else ''
        logger.info(f'✅ {tot_n_spectra} spectra were collected{par_str}.')
        process_time = (time.time() - self.start_process_time) / 60
        logger.info(f'✅ Total compositional analysis time: {process_time:.1f} min')
        print_single_separator()
    
        if is_spectral_quant:
            if is_analysis_successful:
                if is_converged:
                    logger.info('✅ Clustering converged to small errors. All phases identified with confidence higher than 0.8.')
                else:
                    logger.warning('⚠️ Phases could not be identified with confidence higher than 0.8.')
    
                self.print_results()
    
            elif not is_acquisition_successful:
                logger.warning('⚠️ This did not allow to determine which phases are present in the sample.')
            else:
                logger.warning(f'⚠️ Phases could not be identified with the allowed maximum of {self.max_n_spectra} collected spectra.')
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
            logger.info(f"📊 Analysing phases after collection of {tot_n_spectra} spectra...")
    
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
                logger.info("✅ Clustering analysis performed")
    
            # Check whether phase identification converged
            try:
                is_converged = self._is_comp_analysis_converged(max_cl_rmsdist, min_conf)
            except Exception as e:
                raise RuntimeError(f"Error while checking convergence: {e}") from e
    
            if tot_n_spectra >= self.min_n_spectra:
                if is_converged:
                    return is_analysis_successful, is_converged
                elif self.verbose and n_spectra_to_collect > 0:
                    logger.warning("⚠️ Compositional analysis did not converge, more spectra will be collected.")
            elif tot_n_spectra >= self.max_n_spectra:
                logger.info(f"ℹ️ Maximum allowed number of {self.max_n_spectra} was acquired.")
            else:
                if self.verbose:
                    logger.info(f"ℹ️ Collecting additional spectra to reach minimum number of {self.min_n_spectra}.")
    
        elif self.verbose:
            logger.error("❌ Clustering analysis unsuccessful.")
            if n_spectra_to_collect > 0:
                logger.info("ℹ️ More spectra will be collected.")
    
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
            logger.info(f"🔬 Experimental standard acquisition of {self.sample_cfg.ID}")
        
        # Run collection and quantification (fitting optionally performed during collection)
        self._th_peak_energies = {} # Initialise
        self.run_collection_and_quantification(quantify=fit_during_collection)
        
        # Fit standards and save results
        fit_results = self._fit_stds_and_save_results()
        
        # Optionally update the standards library with the new results
        if update_std_library and fit_results is not None and fit_results.lines:
            self._update_standard_library(fit_results)
        
    #%% Save Plots
    def _load_custom_plot_function(self):
        return PlottingModule._load_custom_plot_function(self)

    def _run_custom_clustering_plot(
        self,
        elements: List[str],
        els_comps_list: 'np.ndarray',
        centroids: 'np.ndarray',
        labels: 'np.ndarray',
        els_std_dev_per_cluster: list,
        unused_compositions_list: list
    ) -> bool:
        return PlottingModule._run_custom_clustering_plot(
            self,
            elements,
            els_comps_list,
            centroids,
            labels,
            els_std_dev_per_cluster,
            unused_compositions_list,
        )

    def _save_plots(
        self,
        kmeans: 'KMeans',
        compositions_df: 'pd.DataFrame',
        centroids: 'np.ndarray',
        labels: 'np.ndarray',
        els_std_dev_per_cluster: list,
        unused_compositions_list: list
    ) -> None:
        return PlottingModule._save_plots(
            self,
            kmeans,
            compositions_df,
            centroids,
            labels,
            els_std_dev_per_cluster,
            unused_compositions_list,
        )

    def _save_clustering_plot(
        self,
        elements: List[str],
        els_comps_list: 'np.ndarray',
        centroids: 'np.ndarray',
        labels: 'np.ndarray',
        els_std_dev_per_cluster: list,
        unused_compositions_list: list
    ) -> None:
        return PlottingModule._save_clustering_plot(
            self,
            elements,
            els_comps_list,
            centroids,
            labels,
            els_std_dev_per_cluster,
            unused_compositions_list,
        )

    def _save_violin_plot_powder_mixture(
        self,
        W_mol_frs: List[float],
        ref_names: List[str],
        cluster_ID : int
    ) -> None:
        return PlottingModule._save_violin_plot_powder_mixture(self, W_mol_frs, ref_names, cluster_ID)

    @staticmethod
    def _save_silhouette_plot(
        model: 'KMeans',
        compositions_df: 'pd.DataFrame',
        results_dir: str,
        show_plot: bool
    ) -> None:
        return PlottingModule._save_silhouette_plot(model, compositions_df, results_dir, show_plot)
    
    
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
        labels: Optional[List],
        df_indices: Optional[List],
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
            Dataframe object containing the saved data.
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
            has_composition_result = record is not None and record.composition_atomic_fractions is not None
            has_result = has_composition_result

            if has_result:
                # Unpack spectral quantification results and convert from elemental fraction to % for readability
                atomic_comp = {el + cnst.AT_FR_DF_KEY: round(fr * 100, 2) for el, fr in record.composition_atomic_fractions.items()}
                weight_comp = {el + cnst.W_FR_DF_KEY: round(fr * 100, 2) for el, fr in record.composition_weight_fractions.items()}
                analytical_er = {cnst.AN_ER_DF_KEY: round(float(record.analytical_error) * 100, 2)}

                # Fit quality metrics (present only when a full fit was performed)
                r_squared = None
                redchi_sq = None
                if record.fit_result is not None:
                    r_squared = float(f"{record.fit_result.r_squared:.5f}") if record.fit_result.r_squared is not None else None
                    redchi_sq = float(f"{record.fit_result.reduced_chi_squared:.1f}") if record.fit_result.reduced_chi_squared is not None else None

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

        # Save dataframe
        if data_df is not None and include_spectral_data:
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
        logger.info("📊 Summary of Discarded Spectra")
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
            logger.warning("  ⚠️ Warning: More than 50% of spectra were flagged during quantification!")
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
        logger.info(f"📊 Compositional analysis results for sample {self.sample_cfg.ID}:")
        print_single_separator()
        try:
            logger.info('  Clustering method: %s', self.clustering_cfg.method)
            logger.info('  Clustering features: %s', self.clustering_cfg.features)
            logger.info('  k finding method: %s', self.clustering_cfg.k_finding_method)
            logger.info('  Number of clusters: %d', self.clustering_info[cnst.N_CLUST_KEY])
            logger.info('  WCSS (%%): %.2f', self.clustering_info[cnst.WCSS_KEY] * 10000)
            logger.info('  Silhouette score: %.2f', self.clustering_info[cnst.SIL_SCORE_KEY])
        except KeyError as e:
            raise KeyError(f"Missing key in clustering_info: {e}")
        except AttributeError as e:
            raise AttributeError(f"Missing attribute: {e}")
    
        # Print details on identified phases
        print_single_separator()
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
                phase_table = pd.DataFrame(df_mod_to_print).to_string()
                logger.info("Identified phases:\n%s", phase_table)
        except Exception as e:
            raise RuntimeError(f"Error printing phase results: {e}")
 