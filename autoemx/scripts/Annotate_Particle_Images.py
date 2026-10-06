#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate the particle images of a sample with the positions of the X-ray spectra.

Particle images are saved during acquisition without annotations. This script writes a copy of
each particle image with every spectrum spot drawn as a dot with its spectrum ID, plus a scale bar,
using the spot positions stored in the sample ledger.

Copy this file into a sample folder (the folder containing ledger.json) and run it: the sample is
found automatically. To annotate another sample, set `sample_path` below.

Annotated copies are saved in 'SEM images/annotated'; the original images are not modified.

This script is a thin wrapper around the runner:
autoemx.runners.annotate_particle_images.annotate_particle_images

Created on Mon Oct  5 2026

@author: Andrea

Citation
--------
If you use this package, please cite:
A. Giunto et al., Accurate SEM-EDS Quantification, Automation, and Machine Learning Enable
High-Throughput Compositional Characterization of Powders,
Nature Communications 17, 9735 (2026).
https://doi.org/10.1038/s41467-026-76633-x
"""

# =============================================================================
# Sample Definition
# =============================================================================
import os

sample_path = os.path.dirname(os.path.abspath(__file__)) # Default: the folder containing this script. Replace with the path of a sample folder to annotate another sample.

# =============================================================================
# Options
# =============================================================================
scalebar = True # Draw a scale bar (if the pixel size is stored in the ledger or in the image metadata)
overwrite = True # Overwrite annotated images written by a previous run
output_dir = None # Folder for the annotated images. If None, uses 'SEM images/annotated' in the sample folder

# =============================================================================
# Run
# =============================================================================
from autoemx.runners.annotate_particle_images import annotate_particle_images

annotated_images = annotate_particle_images(
    sample_path=sample_path,
    output_dir=output_dir,
    scalebar=scalebar,
    overwrite=overwrite,
)
