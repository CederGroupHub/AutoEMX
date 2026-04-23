#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for MSA writing and acquisition-time spectrum persistence."""

from pathlib import Path

from autoemx.core.composition_analysis.analyser import EMXSp_Composition_Analyzer
from autoemx.utils import constants as cnst
from autoemx.utils.legacy.spectrum_pointer_writer import write_spectrum_pointer_file


def test_write_spectrum_pointer_file_preserves_template_tail(tmp_path: Path):
    template_lines = [
        "#FORMAT      : EMSA/MAS Spectral Data File\n",
        "#VERSION     : 1.0\n",
        "#NPOINTS     : 2\n",
        "#LIVETIME    : 1.00000000\n",
        "#REALTIME    : 2.00000000\n",
        "#SPECTRUM\n",
        "0,1.0000000000\n",
        "1,2.0000000000\n",
        "#ENDOFDATA   : \n",
        "TRAILER_LINE\n",
    ]
    output_path = tmp_path / "written.msa"

    write_spectrum_pointer_file(
        str(output_path),
        spectrum_vals=[10.0, 20.0, 30.0],
        energy_vals=[0.0, 1.0, 2.0],
        template_lines=template_lines,
        live_time=7.5,
        real_time=8.5,
    )

    written_text = output_path.read_text(encoding="utf-8")

    assert "#ENDOFDATA   :" in written_text
    assert "TRAILER_LINE" in written_text
    assert "0,10.0000000000" in written_text
    assert "1,20.0000000000" in written_text
    assert "2,30.0000000000" in written_text
    assert "0,1.0000000000\n1,2.0000000000\n" not in written_text
