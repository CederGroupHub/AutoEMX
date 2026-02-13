#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Testing of Functions defined in the Electron Microscope Driver
=================================================

This script is designed for manual verification of each function in the
`EM_driver` module of the `autoemxsp` framework. It runs each function once
with safe, example parameters so the tester can visually/ logically verify
correct execution.

IMPORTANT
---------
- Make sure you have the appropriate microscope connection when running
  hardware-specific functions.
- All of the functions below should work properly for automated spectral acquisition with AutoEMXSp.
    None of these is required for spectral quantification and clustering analysis.

Usage
-----
Edit the script to uncomment the desired test function call(s) and run teh script.

Author: Andrea
Created: Mon Oct 13 15:38:36 2025
"""

import numpy as np
from autoemxsp import EM_driver
from autoemxsp.utils.helper import print_single_separator, print_double_separator

#%% -------------------------------------------------------------------------
# Configuration Parameters
# -------------------------------------------------------------------------
microscope_ID = 'PhenomXL'
microscope_type = 'SEM'
detector_type = 'BSD'

voltage = 15
current = 4


# --- Load microscope driver for instrument 'microscope_ID'
try:
    EM_driver.load_microscope_driver(microscope_ID)
except Exception as e:
    raise RuntimeError(f"Failed to load microscope driver: {e}")

#%% -------------------------------------------------------------------------
# Attributes – check they are defined
# -------------------------------------------------------------------------
print_double_separator()
print("Attributes – check they are defined")
print_single_separator()
print("Stage limits (mm):", EM_driver.stage_x_left, EM_driver.stage_x_right,
      EM_driver.stage_y_top, EM_driver.stage_y_bottom)
print(f"Navigation camera image width: {EM_driver.navcam_im_w_mm} mm")
print(f"Offsets of navcam coordinates: ({EM_driver.navcam_x_offset} mm, {EM_driver.navcam_y_offset} mm)")
print(f"Typical working distance: {EM_driver.typical_wd} mm")
print("Default image size [pixels]:", EM_driver.im_width, "x", EM_driver.im_height)
print("is_at_EM flag:", EM_driver.is_at_EM)

# --- Image<->stage coordinate conversion
print_double_separator()
im_height = EM_driver.im_height
im_width = EM_driver.im_width
center_stage = [0.0, 0.0]

print("The microscope coordinate system needs to match the coordinates that are passed to the analyzer object to collect spectral data.")

print_single_separator()
EMcoords_to_pixels_conversion = EM_driver.frame_rel_to_pixel_coords(np.array([center_stage]), im_width, im_height)[0]
print(
    f"Conversion of image coordinates from the microscope coordinate system ({center_stage[0]}, {center_stage[1]}) "
    f"to image pixel coordinates: ({EMcoords_to_pixels_conversion[0]}, {EMcoords_to_pixels_conversion[1]}).\n"
    f"Ensure conversion is correct, considering the image size is {im_width} x {im_height}.",
)

print_single_separator()
pixels_coords_to_EM_conversion = EM_driver.frame_pixel_to_rel_coords(np.array([[int(im_width/2), int(im_height/2)]]), im_width, im_height)[0]
print(
    f"Conversion of image coordinates from pixels ({int(im_width/2)}, {int(im_height/2)}) to the the microscope "
    f"coordinate system: ({pixels_coords_to_EM_conversion[0]}, {pixels_coords_to_EM_conversion[1]}).\n"
    f"Ensure conversion is correct, considering the image size is {im_width} x {im_height}.",
)

#%% -------------------------------------------------------------------------
# Functions to test AT MICROSCOPE
# -------------------------------------------------------------------------
print_double_separator()
# Check if microscope API has been correctly loaded
if not hasattr(EM_driver, "is_at_EM"):
    raise AttributeError("EM_driver.is_at_EM does not exist. Cannot test microscope operation.")

if not EM_driver.is_at_EM:
    raise RuntimeError("EM_driver.is_at_EM is False – microscope connection not available. Cannot test microscope operation.")
else:
    print("✅ EM_driver.is_at_EM exists and is True.") 
    
print("\n========== EM_driver Manual Function Test ==========\n")
# -------------------------------------------------------------------------
# MANUAL FUNCTION CALLS
# Uncomment one block at a time for manual testing
# -------------------------------------------------------------------------

# --- SEM operational controls
# EM_driver.standby()
# EM_driver.set_electron_detector_mode(detector_type)
# EM_driver.activate()
# EM_driver.to_SEM(timeout=5)

# --- SEM beam controls
# EM_driver.set_high_tension(voltage)
# EM_driver.set_beam_current(current)

# --- EDS spectroscopy
# analyzer = EM_driver.get_EDS_analyser_object()
# print("EDS Analyzer:", analyzer)
# print("EDS spectrum:", EM_driver.acquire_XS_spectral_data(analyzer, 0.0, 0.0, 0.5, 10))

# --- Autofocus / image adjustments
# print("Auto focus WD:", EM_driver.auto_focus())
# EM_driver.auto_contrast_brightness()
# EM_driver.adjust_focus(5.0)

# --- Stage motion
# EM_driver.move_to(0.0, 0.0)

# --- Imaging parameters
# print("Current frame width:", EM_driver.get_frame_width())
# print("Range frame width:", EM_driver.get_range_frame_width())
# EM_driver.set_frame_width(1.0)

# --- Acquire image
# img = EM_driver.get_image_data(width=640, height=480)
# print("Image shape:", img.shape)

# --- Adjust brightness/contrast
# EM_driver.set_brightness(0.5)
# EM_driver.set_contrast(0.5)

# --- Working distance
# print("Current WD:", EM_driver.get_current_wd())

# --- NavCam mode
# print("Nav mode activated:", EM_driver.to_nav())
# nav_img = EM_driver.get_navigation_camera_image()
# print("NavCam image shape:", nav_img.shape if nav_img is not None else None)

print("\n✅ Manual function test script loaded. Uncomment calls to run and verify.\n"
      "All of these commands should work properly for automated spectral acquisition with AutoEMXSp. "
      "None is required for spectral quantification and clustering analysis.")