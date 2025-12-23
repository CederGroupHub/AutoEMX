#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoEMXSp default values to define microscope environment.

Defined Variables
------------------
microscope_ID : str, optional
    Identifier for the microscope hardware.
    Must correspond to a calibration folder in `./XSp_calibs/Microscopes/<ID>` (MicroscopeConfig.ID).
    Default is `'PhenomXL'`.
microscope_type : str, optional
    Type of microscope. Allowed: `'SEM'` (implemented), `'STEM'` (not implemented).
    Default is `'SEM'` (MicroscopeConfig.type).
measurement_type : str, optional
    Measurement type. Allowed: `'EDS'` (implemented), `'WDS'` (not implemented).
    Default is `'EDS'` (MeasurementConfig.type).
measurement_mode : str, optional
    Acquisition mode (e.g., `'point'`, `'map'`), defining beam/detector calibration settings.
    Default is `'point'` (MeasurementConfig.mode).
quantification_method : str, optional
    Quantification method. Currently only `'PB'` (Phi-Rho-Z) is implemented.
    Default is `'PB'` (QuantConfig.method).
spectrum_lims : tuple of float, optional
    Lower and upper energy limits for spectrum fitting in eV.
    Default is `(14, 1100)` (QuantConfig.spectrum_lims).
use_instrument_background : bool, optional
    Whether to use instrument background files during fitting.
    If False, background is computed during fitting.
    Default is `False` (QuantConfig.use_instrument_background).

Created on Sun Dec 21 18:59:50 2025

@author: Andrea
"""

microscope_ID: str = 'PhenomXL'

microscope_type: str = 'SEM'

measurement_type: str = 'EDS'

measurement_mode: str = 'point'

quantification_method: str = 'PB'

spectrum_lims: (float, float) = (14, 1100)

use_instrument_background: bool = False

