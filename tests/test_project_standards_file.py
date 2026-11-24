#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standards file chosen for quantification (QuantificationOptionsConfig.project_std_dict_path).

A chosen file is loaded whatever its name and folder, implies project-specific standards, is saved with the
quantification run without being part of its fingerprint, and is required: a missing file is an error, never a silent
fallback to the standards of the microscope. No spectra are quantified.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import autoemx._paths as paths
import autoemx.calibrations as calibs
from autoemx.config.runtime_configs import QuantificationOptionsConfig
from autoemx.config.schema_models import QuantificationConfig
from autoemx.core.composition_analysis.standards import StandardsModule

MICROSCOPE_ID = "PhenomXL"


def _write_standards(path: Path, elements=("Fe",)) -> Path:
    """Standards file with only the lines of ``elements``, copied from the 15 keV file of the microscope."""
    payload = json.loads(paths.user_standards_file(MICROSCOPE_ID, "EDS", 15).read_text())
    point = payload["standards_by_mode"]["point"]
    payload["standards_by_mode"] = {"point": {k: v for k, v in point.items() if k.split("_")[0] in elements}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_config_chosen_file_implies_project_standards(tmp_path):
    cfg = QuantificationOptionsConfig(project_std_dict_path=str(tmp_path / "my_stds.json"))
    assert cfg.use_project_specific_std_dict is True
    assert cfg.project_std_dict_path == str(tmp_path / "my_stds.json")

    assert QuantificationOptionsConfig(project_std_dict_path="").project_std_dict_path is None
    assert QuantificationOptionsConfig().use_project_specific_std_dict is False


def test_load_chosen_file_any_name_and_no_fallback(tmp_path):
    chosen = _write_standards(tmp_path / "elsewhere" / "custom_name.json", elements=("Fe",))

    standards, loaded_path = calibs.load_standards_model("EDS", 15, std_file=str(chosen))
    assert loaded_path == str(chosen)
    assert {k.split("_")[0] for k in standards.standards_by_mode["point"]} == {"Fe"}

    with pytest.raises(FileNotFoundError):
        calibs.load_standards_model("EDS", 15, std_file=str(tmp_path / "missing.json"))


def _analyser_stub(tmp_path: Path, std_file: str) -> SimpleNamespace:
    return SimpleNamespace(
        standards=None,
        sample_result_dir=str(tmp_path / "Sample1"),
        measurement_cfg=SimpleNamespace(mode="point", type="EDS", beam_energy_keV=15),
        exp_stds_cfg=SimpleNamespace(is_exp_std_measurement=False, generate_separate_std_dict=False),
        quant_cfg=QuantificationOptionsConfig(project_std_dict_path=std_file),
    )


def test_analyser_loads_chosen_file(tmp_path):
    chosen = _write_standards(tmp_path / "custom.json", elements=("Fe",))
    # A file of the results folder with the default name must not be used instead
    _write_standards(tmp_path / "EDS_Stds_15keV.json", elements=("Fe", "O"))

    standards, loaded_path = StandardsModule._load_standards(_analyser_stub(tmp_path, str(chosen)))
    assert loaded_path == str(chosen)
    assert {k.split("_")[0] for k in standards.standards_by_mode["point"]} == {"Fe"}


def test_analyser_missing_chosen_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        StandardsModule._load_standards(_analyser_stub(tmp_path, str(tmp_path / "missing.json")))


def test_chosen_file_not_in_fingerprint():
    base = dict(method="PB", energy_range_keV=[0.1, 10.0], fit_tolerance=1e-4, use_instrument_background=False)
    left = QuantificationConfig(quantification_id=0, sample_elements=["Fe"], substrate_elements=[],
                                options={**base, "use_project_specific_std_dict": False,
                                         "project_std_dict_path": None})
    right = QuantificationConfig(quantification_id=0, sample_elements=["Fe"], substrate_elements=[],
                                 options={**base, "use_project_specific_std_dict": True,
                                          "project_std_dict_path": "/a/b.json"})
    assert left.fingerprint() == right.fingerprint()


def test_gui_form_file_parameter(tmp_path):
    from autoemx.gui import backend as be

    chosen = _write_standards(tmp_path / "custom.json", elements=("Fe",))
    values = be.coerce_values({"quant.project_std_dict_path": str(chosen)}, be.QUANT_SPECS)
    assert values["quant.project_std_dict_path"] == str(chosen)
    assert be.quantification_kwargs(values)["project_std_dict_path"] == str(chosen)

    assert be.coerce_values({"quant.project_std_dict_path": "  "}, be.QUANT_SPECS)["quant.project_std_dict_path"] is None
    with pytest.raises(ValueError, match="file not found"):
        be.coerce_values({"quant.project_std_dict_path": str(tmp_path / "missing.json")}, be.QUANT_SPECS)

    acq = be.coerce_values({"aquant.project_std_dict_path": str(chosen)}, be.ACQ_SPECS)
    assert acq["aquant.project_std_dict_path"] == str(chosen)


def test_gui_standards_check_uses_chosen_file(tmp_path):
    from autoemx.gui import backend as be

    chosen = _write_standards(tmp_path / "custom.json", elements=("Fe",))
    assert be.elements_without_standards(["Fe", "O"], MICROSCOPE_ID, 15, std_file=str(chosen)) == ["O"]
    # Project folder: its file with the default name, when present
    _write_standards(tmp_path / "EDS_Stds_15keV.json", elements=("Fe", "O"))
    assert be.elements_without_standards(["Fe", "O"], MICROSCOPE_ID, 15, project_dir=str(tmp_path)) == []
