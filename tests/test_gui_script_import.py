#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loading acquisition scripts in the Acquisition tab of the GUI ("Load script…")."""

from pathlib import Path

import pytest

from autoemx.gui import backend as be
from autoemx.gui.script_import import load_acquisition_script

FORM_KEYS = [s.key for s in be.ACQ_SPECS]
RUN_ACQUISITION = Path(be.__file__).parents[1] / "scripts" / "Run_Acquisition.py"


def _form_values(**changes):
    values = be.coerce_values({s.key: s.default for s in be.ACQ_SPECS}, be.ACQ_SPECS)
    values.update(changes)
    return values


def test_load_run_acquisition_script():
    source = RUN_ACQUISITION.read_text(encoding="utf-8")
    loaded = load_acquisition_script(source, FORM_KEYS)
    assert loaded["mode"] == "eds"
    assert [(r["on"], r["ID"], r["x"], r["y"]) for r in loaded["rows"]] == [(["on"], "Anorthite_mineral", -37.5, -37.5)]
    values = loaded["values"]
    assert values["aacq.beam_energy"] == 15 and values["asample.sample_type"] == "powder"
    assert values["apowder.max_n_par_per_frame"] == 30 and values["abulk.grid_spot_spacing_um"] == 100.0
    # target_Xsp_counts / 10000 * 5: the default time, left empty in the form
    assert values["aacq.max_XSp_acquisition_time"] is None
    # results_dir computed from the location of the script: not loaded
    assert loaded["folder"] is None and loaded["not_loaded"] == ["results_dir"]


def test_commented_samples_are_loaded_unticked():
    source = RUN_ACQUISITION.read_text(encoding="utf-8").replace(
        "'cnd': ['CaAl2Si2O8']},\n]",
        "'cnd': ['CaAl2Si2O8']},\n"
        "    # {'ID': 'Hematite', 'els': ['Fe', 'O'], 'pos': (-22.5, 37.5), 'cnd': []},\n"
        "    # a note, not a sample\n"
        "    #{'ID': 'Spread', 'els': ['Cu'],\n"
        "    #  'pos': (0, 0)},\n"
        "    {'ID': 'Last', 'els': 'Al, O', 'pos': [7.5, -7.5]},\n]")
    rows = load_acquisition_script(source, FORM_KEYS)["rows"]
    assert [(r["ID"], r["on"]) for r in rows] == [
        ("Anorthite_mineral", ["on"]), ("Hematite", []), ("Spread", []), ("Last", ["on"])]
    assert (rows[1]["els"], rows[1]["x"], rows[1]["y"]) == ("Fe, O", -22.5, 37.5)


@pytest.mark.parametrize("mode", ["eds", "eds+sizes", "sizes"])
@pytest.mark.parametrize("quantify", [False, True])
def test_saved_script_loads_back(tmp_path, mode, quantify):
    values = _form_values(**{
        "aacq.beam_energy": 20.0, "aacq.quantify_spectra": quantify, "aacq.n_spectra": 60, "aacq.min_n_spectra": 30,
        "aacq.max_n_spectra": 90, "apowder.max_n_par_per_frame": 12, "asize.n_par_target": 333,
        "asize.min_par_diameter_um": 0.5, "asize.size_stats_ci_tolerance": 5.0})
    rows = [{"on": ["on"], "ID": "A", "els": "Cu, O", "x": -37.5, "y": 37.5, "cnd": "CuO"},
            {"on": [], "ID": "B", "els": "Al, O", "x": 7.5, "y": -7.5, "cnd": ""}]
    samples = be.acquisition_samples(rows, require_elements=mode != "sizes")
    skipped = be.acquisition_samples([{**rows[1], "on": ["on"]}])
    if mode == "sizes":
        kwargs = be.psd_acquisition_kwargs(values)
        samples, skipped = ([{"ID": s["ID"], "pos": s["pos"]} for s in lst] for lst in (samples, skipped))
    else:
        kwargs = be.acquisition_kwargs(values, mode)
    source = be.acquisition_script(samples, kwargs, str(tmp_path), mode, skipped)
    assert "    # {'ID': 'B'" in source
    compile(source, "script", "exec")

    loaded = load_acquisition_script(source, FORM_KEYS)
    assert loaded["mode"] == mode and loaded["folder"] == str(tmp_path) and loaded["not_loaded"] == []
    assert [(r["ID"], r["on"]) for r in loaded["rows"]] == [("A", ["on"]), ("B", [])]
    differing = {k: (values[k], v) for k, v in loaded["values"].items()
                 if values[k] != v and not (k == "asample.els_substrate" and v == ", ".join(values[k]))}
    assert differing == {}


def test_unticked_samples_are_not_acquired():
    rows = [{"on": ["on"], "ID": "A", "els": "Cu", "x": 0, "y": 0, "cnd": ""},
            {"on": [], "ID": "B", "els": "", "x": None, "y": None, "cnd": ""},  # not checked either
            {"ID": "C", "els": "Al", "x": 1, "y": 1, "cnd": ""}]  # rows of older lists: ticked
    assert [s["ID"] for s in be.acquisition_samples(rows)] == ["A", "C"]


@pytest.mark.parametrize("source, error", [
    ("samples = [", "not a valid Python script"),
    ("samples = []\n", "no call to batch_acquire_and_analyze"),
    ("from autoemx.runners.batch_acquire_and_analyze import batch_acquire_and_analyze\n"
     "batch_acquire_and_analyze(samples=make_samples())\n", "the samples must be given as a list"),
])
def test_invalid_scripts(source, error):
    with pytest.raises(ValueError, match=error):
        load_acquisition_script(source, FORM_KEYS)


def test_script_is_not_run(tmp_path):
    marker = tmp_path / "ran"
    source = (f"open({str(marker)!r}, 'w').close()\nsamples = []\n"
              "from autoemx.runners.batch_acquire_and_analyze import batch_acquire_and_analyze\n"
              f"batch_acquire_and_analyze(samples=samples, beam_energy=__import__('os').getpid())\n")
    loaded = load_acquisition_script(source, FORM_KEYS)
    assert not marker.exists() and loaded["not_loaded"] == ["beam_energy"]
