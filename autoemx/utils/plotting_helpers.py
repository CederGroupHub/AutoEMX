#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for user-customizable plotting workflows."""

import os
from importlib import resources

import autoemx.utils.constants as cnst
from .helper import get_sample_dir


def load_custom_plot_template() -> str:
    """Load the packaged custom plot template contents."""
    return resources.files("autoemx").joinpath(
        cnst.CUSTOM_PLOT_TEMPLATE_FILENAME
    ).read_text(encoding="utf-8")


def write_custom_plot_template(custom_plot_file: str, overwrite: bool) -> bool:
    """Write the custom plot template to disk."""
    if os.path.exists(custom_plot_file) and not overwrite:
        return False

    custom_plot_dir = os.path.dirname(custom_plot_file)
    if custom_plot_dir:
        os.makedirs(custom_plot_dir, exist_ok=True)

    with open(custom_plot_file, "w", encoding="utf-8") as f_custom:
        f_custom.write(load_custom_plot_template())
    return True


def refresh_custom_plot_template_file(
    sample_ID: str,
    results_path: str | None = None,
    overwrite: bool = True,
) -> tuple[str, bool]:
    """(Re)create sample-local custom_plot.py from the packaged template."""
    if results_path is None:
        results_path = os.path.join(os.getcwd(), cnst.RESULTS_DIR)

    sample_dir = get_sample_dir(results_path, sample_ID)
    custom_plot_file = os.path.join(sample_dir, cnst.CUSTOM_PLOT_FILENAME)
    was_written = write_custom_plot_template(custom_plot_file, overwrite=overwrite)
    return custom_plot_file, was_written


def ensure_custom_plot_file(sample_dir: str, custom_plot_file: str | None) -> tuple[str, bool]:
    """Ensure a sample-local custom plot file exists and return its path."""
    resolved_custom_plot_file = custom_plot_file
    if not resolved_custom_plot_file:
        resolved_custom_plot_file = os.path.join(sample_dir, cnst.CUSTOM_PLOT_FILENAME)
    elif not os.path.isabs(resolved_custom_plot_file):
        resolved_custom_plot_file = os.path.join(sample_dir, resolved_custom_plot_file)

    was_written = write_custom_plot_template(resolved_custom_plot_file, overwrite=False)
    return resolved_custom_plot_file, was_written
