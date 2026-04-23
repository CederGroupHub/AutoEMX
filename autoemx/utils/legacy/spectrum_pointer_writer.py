#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers to write per-spectrum pointer files with optional vendor template patching."""

import os
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
    energy_vals: Sequence[float],
    *,
    live_time: Optional[float] = None,
    real_time: Optional[float] = None,
) -> None:
    """Write a minimal EMSA-like spectrum file."""
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    n_points = len(spectrum_vals)
    if n_points == 0:
        raise ValueError("Cannot write an empty spectrum pointer file")

    offset = float(energy_vals[0]) if len(energy_vals) > 0 else 0.0
    if len(energy_vals) > 1:
        xperchan = float(energy_vals[1] - energy_vals[0])
    else:
        xperchan = 1.0

    with open(pointer_path, "w", encoding="utf-8") as f:
        f.write("#FORMAT      : EMSA/MAS Spectral Data File\n")
        f.write("#VERSION     : 1.0\n")
        f.write("#NPOINTS     : %d\n" % n_points)
        if live_time is not None:
            f.write("#LIVETIME    : %.8f\n" % float(live_time))
        if real_time is not None:
            f.write("#REALTIME    : %.8f\n" % float(real_time))
        f.write("#OFFSET      : %.8f\n" % offset)
        f.write("#XPERCHAN    : %.8f\n" % xperchan)
        f.write("#SPECTRUM\n")
        for i, count in enumerate(spectrum_vals):
            f.write("%d,%.10f\n" % (i, float(count)))


def write_spectrum_pointer_file(
    pointer_path: str,
    spectrum_vals: List[float],
    energy_vals: Sequence[float],
    *,
    template_lines: Optional[List[str]] = None,
    live_time: Optional[float] = None,
    real_time: Optional[float] = None,
) -> None:
    """Write a spectrum pointer file, preferring vendor template when available."""
    if not template_lines:
        _write_minimal_spectrum_pointer_file(
            pointer_path,
            spectrum_vals,
            energy_vals,
            live_time=live_time,
            real_time=real_time,
        )
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
            for i, count in enumerate(spectrum_vals):
                output_lines.append("%d,%.10f\n" % (i, float(count)))
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
            energy_vals,
            live_time=live_time,
            real_time=real_time,
        )
        return

    with open(pointer_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)