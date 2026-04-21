#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime modules for electron microscopy acquisition and control."""

__all__ = [
    "MicroscopeController",
    "FrameNavigator",
    "SpectrumAcquisition",
    "EM_Sample_Finder",
    "EM_Particle_Finder",
]


def __getattr__(name):
    if name == "MicroscopeController":
        from autoemxsp.core.em_runtime.microscope_controller import MicroscopeController
        return MicroscopeController
    if name == "FrameNavigator":
        from autoemxsp.core.em_runtime.frame_navigator import FrameNavigator
        return FrameNavigator
    if name == "SpectrumAcquisition":
        from autoemxsp.core.em_runtime.spectrum_acquisition import SpectrumAcquisition
        return SpectrumAcquisition
    if name == "EM_Sample_Finder":
        from autoemxsp.core.em_runtime.sample_finder import EM_Sample_Finder
        return EM_Sample_Finder
    if name == "EM_Particle_Finder":
        from autoemxsp.core.em_runtime.particle_finder import EM_Particle_Finder
        return EM_Particle_Finder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
