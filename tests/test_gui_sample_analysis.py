#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the sample-analysis GUI backend and figures (no quantification is run)."""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

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
        for name in model.model_fields:
            assert f"{section}.{name}" in keys


def test_kwargs_match_runner_signatures(results_dir: Path):
    info = be.load_sample_info(str(results_dir / WULFENITE_MINI_ID))
    values = be.coerce_values(be.sample_param_values(info))
    analysis_params = inspect.signature(analyze_sample).parameters
    quant_params = inspect.signature(batch_quantify_and_analyze).parameters
    assert set(be.analysis_kwargs(values)) <= set(analysis_params)
    assert set(be.quantification_kwargs(values)) <= set(quant_params)


def test_coerce_values_reports_all_errors(results_dir: Path):
    info = be.load_sample_info(str(results_dir / WULFENITE_MINI_ID))
    raw = be.sample_param_values(info)
    raw.update({"dbscan.eps": "-1", "clust.ref_formulae": "PbMoO4\nXx2", "quant.els_sample": "Pb, Qq"})
    with pytest.raises(ValueError) as exc:
        be.coerce_values(raw)
    msg = str(exc.value)
    assert "Qq" in msg and "eps" in msg and "Xx2" in msg


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


def test_low_confidence_mixtures_hidden_by_default(powder_dir: Path):
    info = be.load_sample_info(str(powder_dir))
    data = be.load_analysis(info, info.active_analysis)
    data.ref_formulae, data.ref_comps = ["MgO", "Al2O3"], be.reference_compositions(
        ["MgO", "Al2O3"], data.elements, data.features, ["H", "He", "Li"])
    data.mixtures = [[{"refs": ["MgO", "Al2O3"], "conf_score": 0.5}]]

    def n_lines(**kw):
        fig = pl.clustering_figure(data, ["Mg", "Al", "O"], "3d", **kw)
        return sum(1 for t in fig.data if t.legendgroup == "mixtures")

    assert pl.OPT_ZOOM not in pl.DEFAULT_OPTIONS
    assert n_lines() == 0
    assert n_lines(min_mixture_conf=0.4) == 1


def test_create_launcher(tmp_path: Path):
    import os
    import sys

    from autoemx.gui.__main__ import create_launcher

    path = create_launcher(str(tmp_path), results_folder=str(tmp_path))
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
    sections, params = active_params(base, quantify=False)
    assert not sections["quant"] and not sections["dbscan"] and not sections["aitchison"]
    assert sections["merge"] and sections["mixture"] and params["clust.max_k"]
    for geometry in ("aitchison", "auto"):
        assert active_params({**base, "clust.geometry": geometry}, False)[0]["aitchison"]
    sections, params = active_params({**base, "clust.k_forced": 3}, quantify=True)
    assert sections["quant"] and not sections["merge"] and not params["clust.k_finding_method"]
    sections, params = active_params({**base, "clust.method": "dbscan"}, False)
    assert sections["dbscan"] and not sections["merge"] and not params["clust.k_forced"]
    assert not active_params({**base, "clust.do_matrix_decomposition": False}, False)[0]["mixture"]
