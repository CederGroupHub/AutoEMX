#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive GUI to quantify and analyse samples acquired with AutoEMX.

Launch with ``python -m autoemx.gui [results_folder]``. From a script, ``open_acquisition(samples, results_folder)``
opens the Acquisition tab with the samples to acquire filled in, and ``acquisition_report`` /
``wait_for_acquisition`` give the report of the run (see ``autoemx.gui.run_report``).
"""

from autoemx.gui.run_report import acquisition_report, wait_for_acquisition


def open_acquisition(samples, results_folder=None, port=8050, run_id=None, open_browser=True, timeout=120.0):
    """Open the Acquisition tab of the GUI with these samples filled in and return the run ID; see
    ``autoemx.gui.__main__.open_acquisition``."""
    # Imported here: importing __main__ with the package would make ``python -m autoemx.gui`` warn
    from autoemx.gui.__main__ import open_acquisition as _open

    return _open(samples, results_folder, port=port, run_id=run_id, open_browser=open_browser, timeout=timeout)

