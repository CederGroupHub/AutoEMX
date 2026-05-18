#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers to write per-spectrum pointer files with optional vendor template patching."""

import os
import traceback
from typing import List, Optional, Sequence


def load_vendor_msa_template_lines(sample_result_dir: str, template_filename: str) -> Optional[List[str]]:
    """Load vendor-exported MSA template lines from sample root when available."""
    template_path = os.path.join(sample_result_dir, template_filename)
    if not os.path.exists(template_path):
        return None

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.readlines()
    except Exception:
        traceback.print_exc()
        return None


def _replace_msa_header_value(line: str, value: str) -> str:
    """Replace value in a '#KEY: value' MSA header line while preserving the original key."""
    if ":" not in line:
        return line if line.endswith("\n") else line + "\n"
    prefix = line.split(":", maxsplit=1)[0]
    return f"{prefix}: {value}\n"


def _write_minimal_spectrum_pointer_file(
    pointer_path: str,
    spectrum_vals: List[float],
    *,
    xperchan: float,
    offset: float,
    live_time: Optional[float] = None,
    real_time: Optional[float] = None,
) -> None:
    """Write a minimal EMSA-like spectrum file."""
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    n_points = len(spectrum_vals)
    if n_points == 0:
        raise ValueError("Cannot write an empty spectrum pointer file")

    with open(pointer_path, "w", encoding="utf-8") as f:
        f.write("#FORMAT      : EMSA/MAS Spectral Data File\n")
        f.write("#VERSION     : 1.0\n")
        f.write(f"#NPOINTS     : {n_points}\n")
        if live_time is not None:
            f.write(f"#LIVETIME    : {float(live_time):.8f}\n")
        if real_time is not None:
            f.write(f"#REALTIME    : {float(real_time):.8f}\n")
        f.write(f"#OFFSET      : {offset:.3f}\n")
        f.write(f"#XPERCHAN    : {xperchan:.3f}\n")
        f.write("#SPECTRUM\n")
        for i, count in enumerate(spectrum_vals):
            f.write(f"{i},{float(count):.10f}\n")


def write_spectrum_pointer_file(
    pointer_path: str,
    spectrum_vals: List[float],
    *,
    xperchan: float,
    offset: float,
    sample_result_dir: str = None,
    template_filename: str = "EM_metadata.msa",
    live_time: Optional[float] = None,
    real_time: Optional[float] = None,
) -> None:
    """Write a spectrum pointer file (.msa), using vendor template if available, otherwise minimal header. Calibration is always explicit. No index in spectrum data."""
    template_lines = None
    if sample_result_dir is not None and template_filename:
        template_lines = load_vendor_msa_template_lines(sample_result_dir, template_filename)
    if not template_lines:
        # Write minimal EMSA-like spectrum file with explicit calibration
        os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
        n_points = len(spectrum_vals)
        if n_points == 0:
            raise ValueError("Cannot write an empty spectrum pointer file")
        with open(pointer_path, "w", encoding="utf-8") as f:
            f.write("#FORMAT      : EMSA/MAS Spectral Data File\n")
            f.write("#TITLE       : EDS Spectrum\n")
            f.write("#VERSION     : 1.0\n")
            f.write("#OWNER       : Thermo Fisher Scientific Inc.\n")
            f.write(f"#NPOINTS     : {n_points}\n")
            f.write("#NCOLUMNS    : 1\n")
            f.write("#XUNITS      : eV\n")
            f.write("#YUNITS      : Counts\n")
            f.write("#DATATYPE    : Y\n")
            f.write(f"#OFFSET      : {offset:.3f}\n")
            f.write(f"#XPERCHAN    : {xperchan:.3f}\n")
            f.write("#XLABEL      : Energy\n")
            f.write("#YLABEL      : Counts\n")
            f.write("#SIGNALTYPE  : EDS\n")
            f.write("#BEAMKV   -kV: 15.000\n")
            f.write("#AZIMANGLE-dg: 0.0\n")
            f.write("#ELEVANGLE-dg: 28.5\n")
            if live_time is not None:
                f.write(f"#LIVETIME    : {float(live_time):.8f}\n")
            if real_time is not None:
                f.write(f"#REALTIME    : {float(real_time):.8f}\n")
            f.write("#EDSDET      : SDUTW\n")
            f.write("##SPECTRUM    : Spectral Data Starts Here\n")
            for count in spectrum_vals:
                f.write(f"{float(count):.1f}\n")
        return

    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    n_points = len(spectrum_vals)
    if n_points == 0:
        raise ValueError("Cannot write an empty spectrum pointer file")

    output_lines: List[str] = []
    spectrum_section_replaced = False
    preserving_tail = False


    for raw_line in template_lines:
        line = raw_line if raw_line.endswith("\n") else raw_line + "\n"
        stripped = line.strip()
        upper = stripped.upper()

        if not spectrum_section_replaced and upper.startswith("#SPECTRUM"):
            output_lines.append("#SPECTRUM\n")
            for count in spectrum_vals:
                output_lines.append(f"{float(count):.1f}\n")
            spectrum_section_replaced = True
            continue

        if spectrum_section_replaced and not preserving_tail:
            # Skip only the original spectrum data rows. Once the template reaches
            # its post-spectrum footer/header content, preserve the remainder verbatim.
            if stripped.startswith("#"):
                output_lines.append(line)
                preserving_tail = True
            continue

        if preserving_tail:
            output_lines.append(line)
            continue

        if stripped.startswith("#") and ":" in stripped:
            key = stripped[1:].split(":", maxsplit=1)[0].strip()
            key_norm = key.replace("_", "").replace(" ", "").upper()

            if key_norm == "NPOINTS":
                output_lines.append(_replace_msa_header_value(line, str(n_points)))
                continue
            if key_norm == "LIVETIME" and live_time is not None:
                output_lines.append(_replace_msa_header_value(line, f"{float(live_time):.8f}"))
                continue
            if key_norm == "REALTIME" and real_time is not None:
                output_lines.append(_replace_msa_header_value(line, f"{float(real_time):.8f}"))
                continue

        output_lines.append(line)

    if not spectrum_section_replaced:
        _write_minimal_spectrum_pointer_file(
            pointer_path,
            spectrum_vals,
            xperchan=xperchan,
            offset=offset,
            live_time=live_time,
            real_time=real_time,
        )
        return

    with open(pointer_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)