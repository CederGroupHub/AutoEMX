#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Electron Microscopy (EM) Controller and Sample Image Analyzer

This module provides the main EM_Controller class for automated analysis and acquisition in 
scanning electron microscopy (SEM). It acts as a facade/orchestrator that composes lower-level 
modules for microscope control, frame navigation, spectrum acquisition, and image utilities.

Main Classes
------------
EM_Controller
    High-level orchestrator for automated particle detection, mask generation, 
    and X-ray spectra (EDS, WDS) acquisition.

Example Usage
-------------
>>> from autoemxsp.core.em_runtime.controller import EM_Controller
>>> EM_controller = EM_Controller(...)
>>> EM_controller.initialise_SEM()
>>> particle_finder = EM_Particle_Finder(EM_controller, ...)
>>> while particle_finder.go_to_next_particle():
...     xy_spot_list = particle_finder.get_XS_acquisition_spots_coord_list()
...     for (x, y) in xy_spot_list:
...         EM_controller.acquire_XS_spectrum()

Created on Wed Jul 31 09:28:07 2024

@author: Andrea
"""
# Standard library imports
import os
import warnings

# Third-party imports
import cv2
import numpy as np

# Typing
from typing import Any, List, Optional, Tuple, Union

# Local project imports
import autoemxsp.utils.constants as cnst
from autoemxsp.utils import EMError, print_single_separator, draw_scalebar
from autoemxsp.config import (
    MicroscopeConfig,
    SampleConfig,
    MeasurementConfig,
    SampleSubstrateConfig,
    PowderMeasurementConfig,
    BulkMeasurementConfig
)
from autoemxsp import microscope_drivers as EM_driver

# Component modules
from autoemxsp.core.em_runtime.microscope_controller import MicroscopeController
from autoemxsp.core.em_runtime.frame_navigator import FrameNavigator
from autoemxsp.core.em_runtime.spectrum_acquisition import SpectrumAcquisition
from autoemxsp.core.em_runtime import image_utilities


#%% Electron Microscope Controller class    
class EM_Controller:
    """
    High-level orchestrator for automated electron microscopy operations.
    
    This class composes lower-level modules for microscope hardware control,
    frame navigation, spectrum acquisition, and image utilities. It provides
    a unified interface for automated particle detection, X-ray spectra acquisition,
    and sample management.
    
    Configuration is provided via structured dataclasses:
        - MicroscopeConfig
        - SampleConfig
        - MeasurementConfig
        - SampleSubstrateConfig
    
    Main Methods
    ------------
    initialise_SEM()
        Wakes up the SEM microscope, sets measurement parameters.
    initialise_sample_navigator()
        Initializes frame calculation and navigation for the sample.
    initialise_XS_analyzer()
        Gets EDS/WDS analyzer object from EM_driver.
    acquire_XS_spot_spectrum(x, y)
        Acquire X-ray spectral data at position (x,y).
    get_XSp_coords(n_tot_sp_collected)
        Determine next spectrum acquisition location.
    
    Attributes
    ----------
    sample_cfg : SampleConfig
        Sample configuration.
    microscope_cfg : MicroscopeConfig
        Microscope configuration.
    measurement_cfg : MeasurementConfig
        Measurement/acquisition configuration.
    sample_substrate_cfg : SampleSubstrateConfig
        Sample substrate configuration.
    powder_meas_cfg : PowderMeasurementConfig
        Configuration for powder measurement.
    bulk_meas_cfg : BulkMeasurementConfig
        Configuration for bulk measurement.
    init_wd : float
        Initial working distance in mm.
    im_width : int
        Image width in pixels.
    im_height : int
        Image height in pixels.
    pixel_size_um : float or None
        Pixel size in micrometers of current image.
    results_dir : str or None
        Directory for saving result images and data.
    verbose : bool
        If True, print progress and information to the console.
    development_mode (bool)
        Whether class is used for testing without real-time acquisition.
        # Annotation keys for backward compatibility
        an_circle_key = 'circle'
        an_text_key = 'text'
    """
    
    def __init__(
        self,
        microscope_cfg: MicroscopeConfig,
        sample_cfg: SampleConfig,
        measurement_cfg: MeasurementConfig,
        sample_substrate_cfg: SampleSubstrateConfig,
        powder_meas_cfg: PowderMeasurementConfig,
        bulk_meas_cfg: BulkMeasurementConfig,
        init_fw: float = 0.5,
        results_dir: Optional[str] = None,
        verbose: bool = True,
        development_mode: Optional[bool] = False,
    ):
        """
        Initialize an EM_Controller object.
        
        Parameters
        ----------
        microscope_cfg : MicroscopeConfig
            Microscope configuration dataclass instance.
        sample_cfg : SampleConfig
            Sample configuration dataclass instance.
        measurement_cfg : MeasurementConfig
            Measurement/acquisition configuration dataclass instance.
        sample_substrate_cfg : SampleSubstrateConfig
            Sample substrate configuration dataclass instance.
        powder_meas_cfg : PowderMeasurementConfig
            Configuration for powder measurement.
        bulk_meas_cfg : BulkMeasurementConfig
            Configuration for bulk measurement.
        init_fw : float, optional
            Initial frame width in mm for grid search (default: 0.5).
        results_dir : str, optional
            Directory to save result images and data (default: None).
        verbose : bool, optional
            If True, print progress information (default: True).
        development_mode (bool)
            If True, enables offline testing mode (default: False).
        
        Raises
        ------
        RuntimeError
            If the microscope driver cannot be loaded.
        """
        # --- Configuration
        self.sample_cfg = sample_cfg
        self.microscope_cfg = microscope_cfg
        self.measurement_cfg = measurement_cfg
        self.sample_substrate_cfg = sample_substrate_cfg
        self.powder_meas_cfg = powder_meas_cfg
        self.bulk_meas_cfg = bulk_meas_cfg
        
        # --- Load microscope driver
        try:
            EM_driver.load_microscope_driver(microscope_cfg.ID)
            self.EM_driver = EM_driver
        except Exception as e:
            raise RuntimeError(f"Failed to load microscope driver: {e}")
        if not development_mode:
            if not getattr(self.EM_driver, "is_at_EM", False):
                raise EMError("Instrument driver could not be loaded")
        
        # --- Working distance bounds
        if isinstance(measurement_cfg.working_distance, float):
            self.init_wd = measurement_cfg.working_distance
        else:
            self.init_wd = self.EM_driver.typical_wd
        
        # Image dimensions
        self.im_width = self.EM_driver.im_width
        self.im_height = self.EM_driver.im_height
        
        # --- General options
        self.development_mode = development_mode
        self.results_dir = results_dir
        self.verbose = verbose
        
        # --- Variable initializations
        self.is_initialized = False
        self.pixel_size_um: Optional[float] = None
        self.grid_search_fw_mm = init_fw
        
        # --- Initialize component modules
        self.microscope_ctrl = MicroscopeController(
            EM_driver_obj=EM_driver,
            microscope_cfg=microscope_cfg,
            measurement_cfg=measurement_cfg,
            init_wd=self.init_wd,
            im_width=self.im_width,
            im_height=self.im_height,
            verbose=verbose
        )
        
        self.frame_navigator = FrameNavigator(
            EM_controller=self,
            sample_cfg=sample_cfg,
            sample_substrate_cfg=sample_substrate_cfg,
            measurement_cfg=measurement_cfg,
            powder_meas_cfg=powder_meas_cfg,
            bulk_meas_cfg=bulk_meas_cfg,
            center_pos=sample_cfg.center_pos,
            sample_hw_mm=sample_cfg.half_width_mm,
            im_width=self.im_width,
            im_height=self.im_height,
            EM_driver_obj=EM_driver,
            results_dir=results_dir,
            verbose=verbose,
            development_mode=development_mode
        )
        
        self.spectrum_acq = SpectrumAcquisition(
            EM_controller=self,
            sample_cfg=sample_cfg,
            measurement_cfg=measurement_cfg,
            EM_driver_obj=EM_driver,
            im_width=self.im_width,
            im_height=self.im_height,
            results_dir=results_dir,
            verbose=verbose
        )
    
    
    #%% Microscope initialization
    # =============================================================================
    def initialise_SEM(self) -> None:
        """
        Activate and configure the Scanning Electron Microscope (SEM).
        
        Delegates to MicroscopeController which performs:
            1. SEM activation
            2. Detector and beam setup
            3. Working distance adjustment
            4. Focus, brightness, contrast adjustment
        """
        self.microscope_ctrl.initialise_SEM()
        self.is_initialized = True
        
        if self.verbose:
            print("EM_Controller initialization completed.")
    
    
    def initialise_sample_navigator(self, exclude_sample_margin: bool = True) -> None:
        """
        Initialize the sample navigator for automated spectra collection.
        
        Delegates to FrameNavigator which handles frame calculation,
        grid generation, and particle finder initialization.
        
        Parameters
        ----------
        exclude_sample_margin : bool, optional
            Whether to exclude sample margin (default: True).
        """
        self.frame_navigator.initialise_sample_navigator(
            grid_search_fw_mm=self.grid_search_fw_mm,
            exclude_sample_margin=exclude_sample_margin
        )
        
        # Save initial image
        initial_image = self.get_current_image()
        draw_scalebar(initial_image, self.pixel_size_um)
        cv2.imwrite(
            os.path.join(self.results_dir, cnst.INITIAL_SEM_IM_FILENAME + '.png'), 
            initial_image
        )
    
    
    #%% X-ray spectra acquisition
    # =============================================================================
    def initialise_XS_analyzer(self, beam_voltage: float = None) -> None:
        """
        Initialize the EDS analyzer.
        
        Delegates to SpectrumAcquisition module.
        
        Parameters
        ----------
        beam_voltage : float, optional
            Desired beam voltage in kV.
        """
        self.spectrum_acq.initialise_XS_analyzer(beam_voltage=beam_voltage)
    
    
    def get_XSp_coords(
        self, n_tot_sp_collected: int
    ) -> Tuple[bool, Optional[List[Tuple[float, float]]], Optional[int]]:
        """
        Determine X-ray spectrum acquisition coordinates for the next spectrum.
        
        Delegates to SpectrumAcquisition module which handles manual and
        automated spot selection based on measurement mode.
        
        Parameters
        ----------
        n_tot_sp_collected : int
            The current total number of spectra collected.
        
        Returns
        -------
        success : bool
            True if a spot was selected/found, False otherwise.
        spots_xy_list : list of tuple or None
            List of (x, y) coordinates for acquisition, or None if unsuccessful.
        particle_cntr : int or None
            The particle counter/index, or None if not applicable.
        """
        return self.spectrum_acq.get_XSp_coords(
            n_tot_sp_collected,
            self.frame_navigator,
            self.microscope_ctrl
        )
    
    
    def acquire_XS_spot_spectrum(
        self,
        x: float,
        y: float,
        max_acquisition_time: float,
        target_acquisition_counts: int
    ) -> tuple:
        """
        Acquire an X-ray spectrum at the specified position.
        
        Delegates to SpectrumAcquisition module.
        
        Parameters
        ----------
        x, y : float
            X, Y coordinates for spectrum acquisition (normalized coordinates).
        max_acquisition_time : float
            Maximum allowed acquisition time in seconds.
        target_acquisition_counts : int
            Target total X-ray counts.
        
        Returns
        -------
        spectrum_data, background_data, real_time, live_time
        """
        return self.spectrum_acq.acquire_XS_spot_spectrum(
            x, y, max_acquisition_time, target_acquisition_counts
        )
    
    
    #%% Microscope Control
    # =============================================================================
    def adjust_BCF(self) -> float:
        """
        Adjust brightness, contrast, and focus.
        
        Delegates to MicroscopeController.
        
        Returns
        -------
        float
            Timestamp of adjustment completion.
        """
        return self.microscope_ctrl.adjust_BCF()
    
    
    def set_frame_width(self, frame_width: float) -> None:
        """
        Set the frame width at the microscope.
        
        Delegates to MicroscopeController and updates pixel size.
        
        Parameters
        ----------
        frame_width : float
            The desired frame width in millimeters.
        """
        self.microscope_ctrl.set_frame_width(frame_width)
        self.pixel_size_um = frame_width / self.im_width * 1e3
    
    
    def get_current_image(self):
        """
        Acquire image at microscope.
        
        Returns
        -------
        image : np.ndarray
            Image array acquired at the microscope.
        """
        image = self.EM_driver.get_image_data(self.im_width, self.im_height, 1)
        return image
    
    
    def move_to_pos(self, pos: tuple) -> None:
        """
        Move the EM stage to the specified position.
        
        Delegates to MicroscopeController.
        
        Parameters
        ----------
        pos : tuple of float
            Target (x, y) coordinates.
        """
        self.microscope_ctrl.move_to_pos(pos)
    
    
    def convert_pixel_pos_to_mm(self, pos_pixels):
        """
        Convert pixel coordinates to absolute stage coordinates in mm.
        
        Delegates to image_utilities module.
        
        Parameters
        ----------
        pos_pixels : array-like
            Position in pixel coordinates.
        
        Returns
        -------
        pos_abs_mm : ndarray
            Absolute position in millimeters.
        """
        return image_utilities.convert_pixel_pos_to_mm(
            pos_pixels,
            self.im_width,
            self.im_height,
            self.pixel_size_um,
            self.frame_navigator._current_pos if hasattr(self.frame_navigator, '_current_pos') else (0, 0),
            self.EM_driver.image_to_stage_coords_transform
        )
    
    
    def convert_XS_coords_to_pixels(self, xy_coords):
        """
        Convert XS coordinates to pixel coordinates.
        
        Delegates to image_utilities module.
        
        Parameters
        ----------
        xy_coords : tuple
            The XY coordinates in the XS coordinate system.
        
        Returns
        -------
        tuple(int, int)
            Pixel coordinates.
        """
        return image_utilities.convert_XS_coords_to_pixels(
            xy_coords, self.im_width, self.im_height, self.EM_driver
        )
    
    
    @staticmethod
    def standby() -> None:
        """Put microscope in standby mode."""
        MicroscopeController.standby()
    
    
    #%% Frame Navigation
    # =============================================================================
    def go_to_next_frame(self) -> bool:
        """
        Move the microscope to the next frame position.
        
        Delegates to FrameNavigator.
        
        Returns
        -------
        bool
            True if moved to next frame, False if no frames remain.
        """
        success = self.frame_navigator.go_to_next_frame(self.microscope_ctrl)
        # Update pixel size if frame width changed
        try:
            frame_width_mm = self.EM_driver.get_frame_width()
            self.pixel_size_um = frame_width_mm / self.im_width * 1e3
        except:
            pass
        return success
    
    
    def save_frame_image(self, filename, im_annotations=None, scalebar=True, 
                         frame_image=None, save_dir=None):
        """
        Save an annotated and raw EM frame as multi-page TIFF.
        
        Delegates to image_utilities module.
        
        Parameters
        ----------
        filename : str
            Name used for saved .tif image file.
        im_annotations : dict | list(dict) | None, optional
            Dictionary with annotations ('text', 'circle' keys).
        scalebar : bool, optional
            Whether to annotate with a scalebar (default: True).
        frame_image : np.ndarray | None, optional
            Frame image to save. Acquires current if not provided.
        save_dir : str, optional
            Directory to save the TIFF file.
        """
        if not save_dir:
            save_dir = self.results_dir
        
        image_utilities.save_frame_image(
            frame_image=frame_image,
            pixel_size_um=self.pixel_size_um,
            im_width=self.im_width,
            im_height=self.im_height,
            sample_cfg=self.sample_cfg,
            microscope_cfg=self.microscope_cfg,
            filename=filename,
            results_dir=save_dir,
            im_annotations=im_annotations,
            scalebar=scalebar,
            EM_driver=self.EM_driver,
            auto_adjust_bc=self.microscope_cfg.is_auto_BC
        )
    
    
    # Expose frame navigator attributes for backward compatibility
    @property
    def frame_pos_mm(self):
        """Frame positions in mm."""
        return self.frame_navigator.frame_pos_mm
    
    @property
    def frame_labels(self):
        """Frame labels."""
        return self.frame_navigator.frame_labels
    
    @property
    def num_frames(self):
        """Number of frames."""
        return self.frame_navigator.num_frames
    
    @property
    def current_frame_label(self):
        """Current frame label."""
        return self.frame_navigator.current_frame_label
    
    @property
    def particle_finder(self):
        """Particle finder instance."""
        return self.frame_navigator.particle_finder
