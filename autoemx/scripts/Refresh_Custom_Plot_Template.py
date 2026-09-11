#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate sample-local custom_plot.py from the packaged template.

Citation
--------
If you use this package, please cite:
A. Giunto et al., Accurate SEM-EDS Quantification, Automation, and Machine Learning Enable
High-Throughput Compositional Characterization of Powders,
Nature Communications 17, 9735 (2026).
https://doi.org/10.1038/s41467-026-76633-x
"""

from autoemx.runners import refresh_custom_plot_template

# =============================================================================
# Sample Definition
# =============================================================================
sample_ID = 'Wulfenite_example'
results_path = None  # Looks in default Results folder if left unspecified
overwrite_existing_file = True

# =============================================================================
# Run
# =============================================================================
custom_plot_path = refresh_custom_plot_template(
    sample_ID=sample_ID,
    results_path=results_path,
    overwrite=overwrite_existing_file,
)

print(f"Custom plot template path: {custom_plot_path}")
