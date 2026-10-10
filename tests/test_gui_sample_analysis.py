#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the sample-analysis GUI backend and figures (no quantification is run)."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

import autoemx.utils.constants as cnst

pytest.importorskip("plotly")

from autoemx.config.ledger_schemas import (
    AitchisonParams,
    ClusterMergeParams,
    DBSCANParams,
    MixtureParams,
)
from autoemx.gui import backend as be
from autoemx.gui import plots as pl
from autoemx.runners.analyze_sample import analyze_sample
from autoemx.runners.batch_quantify_and_analyze import batch_quantify_and_analyze
from ci_paths import INPUTS_DIR, K412_CLUSTER_MINI_ID, TESTS_DIR, WULFENITE_MINI_ID

POWDER_EXAMPLE = TESTS_DIR.parent / "examples" / "Results" / "Powder_mixture_example"


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    for sample in (WULFENITE_MINI_ID, K412_CLUSTER_MINI_ID):
        shutil.copytree(INPUTS_DIR / sample, tmp_path / sample)
    return tmp_path


@pytest.fixture()
def powder_dir(tmp_path: Path) -> Path:
    if not (POWDER_EXAMPLE / "spectra").is_dir():
        pytest.skip("Powder_mixture_example spectra not available")
    dest = tmp_path / "res" / POWDER_EXAMPLE.name
    shutil.copytree(POWDER_EXAMPLE, dest)
    return dest


def test_find_samples(results_dir: Path):
    samples = be.find_samples(str(results_dir))
    assert [s.sample_id for s in samples] == [K412_CLUSTER_MINI_ID, WULFENITE_MINI_ID]
    with pytest.raises(FileNotFoundError):
        be.find_samples(str(results_dir / "missing"))


def test_param_specs_cover_every_clustering_submodel_field():
    keys = set(be.SPECS_BY_KEY)
    for section, model in (("dbscan", DBSCANParams), ("aitchison", AitchisonParams),
                           ("merge", ClusterMergeParams), ("mixture", MixtureParams)):
        for name in be._SUBMODEL_FIELDS.get(section) or model.model_fields:
            assert f"{section}.{name}" in keys
    # Mixture decomposition: only the main parameters are exposed
    assert {k for k in keys if k.startswith("mixture.")} == {
        "mixture.max_n_phases", "mixture.collapse_equivalent_mixtures", "mixture.equivalent_span_tol"}


def test_kwargs_match_runner_signatures(results_dir: Path):
    from autoemx.runners.fit_and_quantify_spectrum_from_ledger import fit_and_quantify_spectrum_from_ledger

    info = be.load_sample_info(str(results_dir / WULFENITE_MINI_ID))
    values = be.coerce_values(be.sample_param_values(info))
    assert set(be.analysis_kwargs(values)) <= set(inspect.signature(analyze_sample).parameters)

    qvalues = be.coerce_values({s.key: s.default for s in be.QUANT_SPECS}, be.QUANT_SPECS)
    qkwargs = be.quantification_kwargs(qvalues)
    assert set(qkwargs) <= set(inspect.signature(batch_quantify_and_analyze).parameters)
    # Empty / 'saved' options keep each sample's saved value
    assert qkwargs["spectrum_lims"] is None and qkwargs["use_project_specific_std_dict"] is None
    assert "use_instrument_background" not in qkwargs and qkwargs["run_analysis"] is True
    assert qkwargs["max_spectra_to_quantify"] is None

    svalues = be.coerce_values(be.single_param_values(info), be.SINGLE_SPECS)
    skwargs = be.single_fit_kwargs(svalues)
    skwargs.pop("quantify")
    assert set(skwargs) <= set(inspect.signature(fit_and_quantify_spectrum_from_ledger).parameters)
    assert skwargs["els_sample"] == ["Pb", "Mo", "O"]


def test_coerce_values_reports_all_errors(results_dir: Path):
    info = be.load_sample_info(str(results_dir / WULFENITE_MINI_ID))
    raw = be.sample_param_values(info)
    raw.update({"dbscan.eps": "-1", "clust.ref_formulae": "PbMoO4\nXx2"})
    with pytest.raises(ValueError) as exc:
        be.coerce_values(raw)
    msg = str(exc.value)
    assert "eps" in msg and "Xx2" in msg
    single = be.single_param_values(info)
    single.update({"single.els_sample": "Pb, Qq", "single.std_formula": "Zz3"})
    with pytest.raises(ValueError) as exc:
        be.coerce_values(single, be.SINGLE_SPECS)
    assert "Qq" in str(exc.value) and "Zz3" in str(exc.value)


def test_k_forced_mapping(results_dir: Path):
    info = be.load_sample_info(str(results_dir / WULFENITE_MINI_ID))
    values = be.coerce_values({**be.sample_param_values(info), "clust.k_forced": 3})
    kwargs = be.analysis_kwargs(values)
    assert kwargs["k_forced"] == 3 and kwargs["k_finding_method"] is None
    values = be.coerce_values({**be.sample_param_values(info), "clust.k_forced": None})
    kwargs = be.analysis_kwargs(values)
    assert kwargs["k_forced"] is False and kwargs["k_finding_method"] == values["clust.k_finding_method"]


def test_analysis_and_figures(powder_dir: Path):
    info = be.load_sample_info(str(powder_dir))
    values = be.coerce_values(be.sample_param_values(info))
    values["clust.ref_formulae"] = ["MgO", "Al2O3", "MgAl2O4"]
    values["clust.k_forced"] = 2
    analyzer = analyze_sample(
        sample_ID=powder_dir.name, results_path=str(powder_dir.parent), **be.analysis_kwargs(values)
    )
    assert analyzer is not None

    info = be.load_sample_info(str(powder_dir))
    data = be.load_analysis(info, info.active_analysis)
    assert data.summary["n_clusters"] == 2
    assert data.summary["n_clustered"] == sum(data.n_points)
    assert set(data.comps["status"]) <= {"clustered", "discarded", "not quantified"}
    assert list(data.ref_comps.index) == ["MgO", "Al2O3", "MgAl2O4"]
    assert data.ref_comps.loc["MgAl2O4", "Mg"] == pytest.approx(1 / 7)

    sid = data.comps.loc[data.comps["status"] == "clustered", "spectrum"].iloc[0]
    for mode, axes in (("3d", ["Mg", "Al", "O"]), ("ternary", ["Mg", "Al", "O"]), ("2d", ["Mg", "Al"])):
        fig = pl.clustering_figure(data, axes, mode, highlight=sid)
        names = {t.name for t in fig.data}
        assert {"Cluster 0", "Centroids", "Candidate phases", f"Spectrum {sid}"} <= {
            n.split(" (")[0] if n else n for n in names
        }
    assert pl.distribution_figure(data).data

    raw = be.load_raw_spectrum(info, sid)
    assert len(raw["energy"]) == len(raw["counts"]) > 0
    assert pl.spectrum_figure(raw).data


def test_clustering_zoom(powder_dir: Path):
    info = be.load_sample_info(str(powder_dir))
    data = be.load_analysis(info, info.active_analysis)
    assert [o["value"] for o in pl.zoom_options(None)] == [pl.ZOOM_FULL, pl.ZOOM_DATA]
    options = pl.zoom_options(data)
    assert [o["value"] for o in options][2:] == [f"{pl.ZOOM_CLUSTER}{i}" for i in range(len(data.centroids))]
    assert options[2]["label"] == f"Cluster 0 ({data.n_points[0]})"

    axes = ["Mg", "Al", "O"]
    members = data.comps[(data.comps["status"] == "clustered") & (data.comps["cluster"] == 0)]
    pts = members[axes].to_numpy(dtype=float) * 100
    spans = {}
    for zoom in (pl.ZOOM_FULL, pl.ZOOM_DATA, f"{pl.ZOOM_CLUSTER}0"):
        scene = pl.clustering_figure(data, axes, "3d", zoom=zoom).layout.scene
        ranges = [sorted(scene[f"{a}axis"].range) for a in "xyz"]
        assert all(lo <= pts[:, i].min() and pts[:, i].max() <= hi for i, (lo, hi) in enumerate(ranges))
        spans[zoom] = [hi - lo for lo, hi in ranges]
    assert spans[pl.ZOOM_FULL] == [106, 106, 106]
    assert all(c < d <= f for c, d, f in zip(spans[f"{pl.ZOOM_CLUSTER}0"], spans[pl.ZOOM_DATA],
                                             spans[pl.ZOOM_FULL]))

    # Ternary: the zoomed triangle contains the (normalised) cluster; 2D: the axes contain it
    tern = pl.clustering_figure(data, axes, "ternary", zoom=f"{pl.ZOOM_CLUSTER}0").layout.ternary
    norm = pts / pts.sum(axis=1, keepdims=True) * 100
    mins = [tern.baxis.min, tern.caxis.min, tern.aaxis.min]  # b: x, c: y, a: z (as in _point_trace)
    assert sum(mins) > 0 and all(m <= norm[:, i].min() for i, m in enumerate(mins))
    lay = pl.clustering_figure(data, axes[:2], "2d", zoom=f"{pl.ZOOM_CLUSTER}0").layout
    for i, ax in enumerate((lay.xaxis, lay.yaxis)):
        assert ax.range[0] <= pts[:, i].min() and pts[:, i].max() <= ax.range[1] and ax.range[1] - ax.range[0] < 100
    # Unknown cluster: zoomed on all spectra
    assert pl.clustering_figure(data, axes, "3d", zoom=f"{pl.ZOOM_CLUSTER}99").layout.scene.xaxis.range == \
        pl.clustering_figure(data, axes, "3d", zoom=pl.ZOOM_DATA).layout.scene.xaxis.range


def test_low_confidence_mixtures_hidden_by_default(powder_dir: Path):
    info = be.load_sample_info(str(powder_dir))
    data = be.load_analysis(info, info.active_analysis)
    data.ref_formulae, data.ref_comps = ["MgO", "Al2O3"], be.reference_compositions(
        ["MgO", "Al2O3"], data.elements, data.features, ["H", "He", "Li"])
    data.mixtures = [[{"refs": ["MgO", "Al2O3"], "conf_score": 0.5}]]

    def n_lines(**kw):
        fig = pl.clustering_figure(data, ["Mg", "Al", "O"], "3d", **kw)
        return sum(1 for t in fig.data if t.legendgroup == "mixtures")

    assert n_lines() == 0
    assert n_lines(min_mixture_conf=0.4) == 1


def test_create_launcher(tmp_path: Path):
    import os
    import sys

    from autoemx.gui.__main__ import create_launcher

    ext = ".bat" if os.name == "nt" else ".command" if sys.platform == "darwin" else ".sh"
    old_launcher = tmp_path / f"AutoEMX GUI{ext}"
    old_launcher.write_text('python -m autoemx.gui\n')
    other_file = tmp_path / f"AutoEMX GUI notes{ext}"
    other_file.write_text("not a launcher")

    path = create_launcher(str(tmp_path), results_folder=str(tmp_path))
    assert path.name == f"AutoEMX{ext}"
    assert not old_launcher.exists() and other_file.exists()  # only the old launcher is replaced
    text = path.read_text()
    assert sys.executable in text and "-m autoemx.gui" in text and str(tmp_path) in text
    if os.name != "nt":
        assert os.access(path, os.X_OK)


def test_sem_images_linked_to_spectra(tmp_path: Path):
    from PIL import Image

    images_dir = tmp_path / "S" / "SEM images"
    images_dir.mkdir(parents=True)
    for name in ("S_par10_frE1_xyspots.png", "S_par2_frE1_xyspots.png", "S_frE1_particles.png",
                 "S_fr3_xyspots.tif", "S_par2_frE1_xyspots_raw.png"):
        Image.new("RGB", (40, 30)).save(images_dir / name)
    images = be.find_sem_images(str(tmp_path / "S"))
    assert [im.name for im in images] == [
        "S_fr3_xyspots.tif", "S_par2_frE1_xyspots.png", "S_par10_frE1_xyspots.png", "S_frE1_particles.png"]
    assert images[3].kind == "frame" and images[3].label == "Frame E1"
    assert be.sem_image_for_spectrum(images, 10, "E1").name == "S_par10_frE1_xyspots.png"
    assert be.sem_image_for_spectrum(images, None, "3").name == "S_fr3_xyspots.tif"
    assert be.sem_image_for_spectrum(images, 7, "E1") is None
    png, size = be.read_image_png(images[0].path, 20)
    assert size == (40, 30) and png[:4] == b"\x89PNG"

    comps = be.pd.DataFrame({"spectrum": ["0", "1", "2"], "particle": [2, 10, 10], "frame": ["E1"] * 3,
                             "px": [5, 6, None], "py": [5, 6, 7]})
    assert be.spectra_on_image(comps, images[2])["spectrum"].tolist() == ["1"]


def test_load_analysis_without_clustering(results_dir: Path):
    info = be.load_sample_info(str(results_dir / K412_CLUSTER_MINI_ID))
    data = be.load_analysis(info, None)
    assert len(data.comps) == info.n_spectra
    assert data.summary["n_clusters"] == 0
    fig = pl.clustering_figure(data, data.elements[:3], "3d")
    assert fig.data


def test_active_params():
    from autoemx.gui.app import active_params

    base = {"clust.method": "kmeans", "clust.geometry": "euclidean", "clust.k_forced": None,
            "clust.auto_merge_clusters": True, "clust.do_matrix_decomposition": True}
    sections, params = active_params(base)
    assert not sections["dbscan"] and not sections["aitchison"]
    assert sections["merge"] and sections["mixture"] and params["clust.max_k"]
    for geometry in ("aitchison", "auto"):
        assert active_params({**base, "clust.geometry": geometry})[0]["aitchison"]
    sections, params = active_params({**base, "clust.k_forced": 3})
    assert not sections["merge"] and not params["clust.k_finding_method"]
    sections, params = active_params({**base, "clust.method": "dbscan"})
    assert sections["dbscan"] and not sections["merge"] and not params["clust.k_forced"]
    assert not active_params({**base, "clust.do_matrix_decomposition": False})[0]["mixture"]


def test_sample_summary_and_quant_progress(results_dir: Path):
    summary = be.sample_summary(str(results_dir / K412_CLUSTER_MINI_ID))
    assert summary["sample"] == K412_CLUSTER_MINI_ID and summary["n_spectra"] == 6
    assert summary["elements"] == "Fe, Mg, Ca, Al, Si, O" and len(summary["date"]) == 16
    runs = be.quantification_runs(be.load_sample_info(str(results_dir / K412_CLUSTER_MINI_ID)))
    assert runs and runs[-1]["n_spectra"] == 6

    log = (f"{be.QUANT_SAMPLE_MARKER} 1/2: A\nINFO: Starting quantification of 3 spectra on up to 6 cores.\n"
           f" Spectrum #1/5:\n  x\n Spectrum #0/5:\n\n{be.QUANT_SAMPLE_MARKER} 2/2: B\n"
           "INFO: Starting quantification of 4 spectra\n Spectrum #3/9:\n")
    progress = be.quant_progress(log, ["A", "B"])
    assert progress[0] == {"sample": "A", "state": "done", "done": 2, "total": 3}
    assert progress[1] == {"sample": "B", "state": "running", "done": 1, "total": 4}


def test_acquisition_settings():
    from autoemx.gui.tab_acquisition import active_acq_params
    from autoemx.runners.batch_acquire_and_analyze import batch_acquire_and_analyze

    defaults = {s.key: s.default for s in be.ACQ_SPECS}
    values = be.coerce_values(defaults, be.ACQ_SPECS)
    kwargs = be.acquisition_kwargs(values)
    assert set(kwargs) <= set(inspect.signature(batch_acquire_and_analyze).parameters)
    assert kwargs["max_XSp_acquisition_time"] == 25  # 50000 counts / 10000 * 5 s, as in Run_Acquisition.py
    assert kwargs["els_substrate"] == ["C"]
    # Without quantification, "number of spectra" sets the spectra collected; with it, min and max
    # (also as the minimum: the analyser raises max_n_spectra to min_n_spectra)
    no_quant = be.acquisition_kwargs({**values, "aacq.n_spectra": 5})
    assert (no_quant["min_n_spectra"], no_quant["max_n_spectra"]) == (5, 5)
    quant = be.acquisition_kwargs({**values, "aacq.quantify_spectra": True, "aacq.n_spectra": 5})
    assert (quant["min_n_spectra"], quant["max_n_spectra"]) == (50, 100)
    assert kwargs["powder_meas_cfg_kwargs"]["max_area_par"] == 10000.0
    assert "par_spot_selection_mode" not in kwargs["powder_meas_cfg_kwargs"]

    samples = be.acquisition_samples([
        {"ID": "Anorthite", "els": "Ca, Al, Si, O", "x": "-37.5", "y": -37.5, "cnd": "CaAl2Si2O8"},
        {"ID": "", "els": "", "x": None, "y": None, "cnd": ""},  # empty rows are ignored
    ])
    assert samples == [{"ID": "Anorthite", "els": ["Ca", "Al", "Si", "O"], "pos": (-37.5, -37.5),
                        "cnd": ["CaAl2Si2O8"]}]
    with pytest.raises(ValueError) as exc:
        be.acquisition_samples([{"ID": "a/b", "els": "Qq", "x": "s", "y": 1, "cnd": "Xx2"},
                                {"ID": "a/b", "els": "O", "x": 0, "y": 0}])
    assert all(word in str(exc.value) for word in ("cannot contain", "Qq", "numbers", "Xx2", "duplicated"))

    script = be.acquisition_script(samples, kwargs, "/data/results")
    compile(script, "Run_Acquisition_GUI.py", "exec")
    assert "batch_acquire_and_analyze(" in script and "'Anorthite'" in script

    active = active_acq_params(defaults)
    assert (active["apowder"], active["abulk"], active["aquant"]) == ("on", "off", "off")
    assert (active["aacq.n_spectra"], active["aacq.min_n_spectra"], active["aacq.contrast"]) == ("on", "hidden", "hidden")
    active = active_acq_params({**defaults, "asample.sample_type": "bulk", "aacq.quantify_spectra": True,
                                "aacq.auto_adjust_brightness_contrast": False})
    assert (active["apowder"], active["abulk"], active["aquant"]) == ("off", "on", "on")
    assert (active["aacq.n_spectra"], active["aacq.max_n_spectra"], active["aacq.brightness"]) == ("hidden", "on", "on")

    log = "Sample 'Anorthite'\n🔬 Acquiring spectrum #0...\n🔬 Acquiring spectrum #1...\nSample 'B'\n🔬 Acquiring spectrum #2..."
    progress = be.acquisition_progress(log, ["Anorthite", "B", "C"], 100)
    assert [(p["state"], p["done"]) for p in progress] == [("done", 2), ("running", 1), ("waiting", 0)]
    log = ("Sample 'A'\n🔬 Acquiring spectrum #0...\n12:00:00 ERROR: Sample 'A': acquisition/quantification failed: "
           "no particles found\nTraceback...\nSample 'B'\n🔬 Acquiring spectrum #0...")
    progress = be.acquisition_progress(log, ["A", "B"], 100)
    assert [(p["state"], p["error"]) for p in progress] == [("failed", "no particles found"), ("running", None)]


def test_import_spectra_folder(tmp_path: Path):
    source = INPUTS_DIR / WULFENITE_MINI_ID / cnst.SPECTRA_DIR
    scan = be.inspect_spectra_folder(str(source))
    n_files = len([p for p in source.iterdir() if p.suffix.lower() in cnst.EMSA_SPECTRUM_EXTENSIONS])
    assert scan["n_files"] == n_files > 0 and scan["beam_energies"] == [15.0] and scan["calibration"]

    results = tmp_path / "results"
    results.mkdir()
    (results / "taken").mkdir()
    args = dict(folder=str(source), results_dir=str(results), elements="Pb, Mo, O", substrate="C, O, Al",
                sample_type="powder", microscope_id=be.dflt.microscope_ID, beam_energy=15)
    for bad, msg in ((dict(sample_id="taken"), "already exists"), (dict(sample_id="a b"), "sample ID"),
                     (dict(sample_id="ok", elements=""), "elements"), (dict(sample_id="ok", beam_energy=None), "beam")):
        with pytest.raises(ValueError, match=msg):
            be.import_kwargs(**{**args, **bad})

    kwargs = be.import_kwargs(sample_id="Imported", **args)
    be._run_import("", {"kwargs": kwargs})
    info = be.load_sample_info(str(results / "Imported"))
    assert info.n_spectra == n_files and info.elements == ["Pb", "Mo", "O"]
    cfg = info.ledger.configs
    assert (cfg.microscope_cfg.energy_zero, cfg.microscope_cfg.bin_width) == pytest.approx(scan["calibration"])
    assert cfg.sample_cfg.type == "powder" and cfg.measurement_cfg.beam_energy_keV == 15
    assert len(list(source.iterdir())) == n_files  # source untouched
    # Failed import: no partial sample folder left
    with pytest.raises(Exception):
        be._run_import("", {"kwargs": {**kwargs, "samples": [{**kwargs["samples"][0], "ID": "Empty",
                                                                 "spectra_dir": str(tmp_path)}]}})
    assert not (results / "Empty").exists()


def test_import_requires_one_energy_calibration(tmp_path: Path):
    source = INPUTS_DIR / WULFENITE_MINI_ID / cnst.SPECTRA_DIR
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in cnst.EMSA_SPECTRUM_EXTENSIONS)[:2]
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    for i, f in enumerate(files):
        text = f.read_text(encoding="utf-8", errors="replace")
        if i:  # second file: another channel width
            text = "\n".join("#XPERCHAN    : 5.0" if line.upper().startswith("#XPERCHAN") else line
                              for line in text.splitlines())
        (mixed / f.name).write_text(text, encoding="utf-8")
    scan = be.inspect_spectra_folder(str(mixed))
    assert scan["calibration"] is None and "different energy calibrations" in scan["calibration_error"]

    results = tmp_path / "results"
    results.mkdir()
    kwargs = be.import_kwargs(str(mixed), str(results), "Mixed", "Pb, Mo, O", "C", "powder",
                              be.dflt.microscope_ID, 15)
    with pytest.raises(RuntimeError):
        be._run_import("", {"kwargs": kwargs})
    assert not (results / "Mixed").exists()


def test_windows_launcher_shortcut():
    from PIL import Image

    from autoemx.gui.__main__ import LAUNCHER_ICON_WINDOWS, _windows_shortcut_script

    # Windows icon with the sizes shown on the Desktop, up to 256 px
    assert (256, 256) in Image.open(LAUNCHER_ICON_WINDOWS).info["sizes"]
    script = _windows_shortcut_script(Path("C:/Users/O'Neil/Desktop/AutoEMX.lnk"), Path("C:/x/AutoEMX_1.bat"),
                                      LAUNCHER_ICON_WINDOWS, Path("C:/Users/O'Neil"))
    assert "'C:/Users/O''Neil/Desktop/AutoEMX.lnk'" in script.replace("\\", "/")  # quotes escaped
    assert f"{LAUNCHER_ICON_WINDOWS},0" in script and script.endswith("$s.Save()")


def test_remove_unfinished_sample(tmp_path: Path):
    unfinished = tmp_path / "unfinished"
    (unfinished / cnst.SPECTRA_DIR).mkdir(parents=True)
    finished = tmp_path / "finished"
    finished.mkdir()
    (finished / be.LEDGER_NAME).write_text("{}", encoding="utf-8")
    for path in (unfinished, finished, tmp_path / "missing", None):
        be.remove_unfinished_sample(str(path) if path else None)
    assert not unfinished.exists() and finished.exists()
