#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for Aitchison-geometry clustering support in composition analysis."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autoemx.config.ledger_schemas import AitchisonParams, ClusteringAnalysis, ClusteringConfig, DBSCANParams
from autoemx.core.composition_analysis.clustering import ClusteringModule


# --- Config model -----------------------------------------------------------
def test_clustering_config_geometry_defaults_to_auto():
    assert ClusteringConfig().geometry == "auto"


def test_saved_clustering_config_without_geometry_is_backfilled_as_euclidean():
    """Ledgers saved before the geometry field existed were clustered in Euclidean geometry."""
    legacy = ClusteringAnalysis.model_validate({"config": {"method": "kmeans", "features": "at_fr"}})
    assert legacy.config.geometry == "euclidean"
    # Explicitly saved geometries are preserved
    saved = ClusteringAnalysis.model_validate({"config": {"method": "kmeans", "geometry": "auto"}})
    assert saved.config.geometry == "auto"
    # Configs built in code keep the new default
    assert ClusteringAnalysis(config=ClusteringConfig()).config.geometry == "auto"


def test_backfilled_config_keeps_pre_geometry_fingerprint():
    payload = {"method": "kmeans", "features": "at_fr", "k_finding_method": "silhouette"}
    legacy = ClusteringAnalysis.model_validate({"config": payload}).config
    assert "geometry" not in legacy.fingerprint_payload()


def test_legacy_config_loader_sets_euclidean_geometry(tmp_path):
    import json
    from autoemx.utils.legacy.ledger_bootstrap import _load_legacy_clustering_cfg
    (tmp_path / "Comp_analysis_configs.json").write_text(json.dumps({"clustering_cfg": {"method": "kmeans", "k": 2}}))
    cfg = _load_legacy_clustering_cfg(str(tmp_path))
    assert cfg is not None and cfg.geometry == "euclidean"


def test_clustering_config_geometry_is_normalized():
    assert ClusteringConfig(geometry=" Aitchison ").geometry == "aitchison"


def test_clustering_config_rejects_unknown_geometry():
    with pytest.raises(ValueError):
        ClusteringConfig(geometry="manhattan")


# --- Fingerprint behavior ---------------------------------------------------
def test_euclidean_fingerprint_excludes_geometry_block():
    """Existing configs must keep their fingerprint stable (no geometry keys)."""
    payload = ClusteringConfig(geometry="euclidean").fingerprint_payload()
    assert "geometry" not in payload
    assert "aitchison" not in payload


def test_aitchison_fingerprint_differs_and_reacts_to_detection_limit():
    euclidean = ClusteringConfig(geometry="euclidean")
    base = ClusteringConfig(geometry="aitchison")
    changed = ClusteringConfig(geometry="aitchison", aitchison=AitchisonParams(detection_limit_percent=1.0))
    assert base.fingerprint() != euclidean.fingerprint()
    assert base.fingerprint() != changed.fingerprint()


# --- Transforms -------------------------------------------------------------
def test_zero_replacement_is_closed_positive_and_preserves_ratios():
    X = np.array([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5]])
    out = ClusteringModule._multiplicative_replacement(X, 0.005)
    assert np.all(out > 0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert out[0, 2] == pytest.approx(0.65 * 0.005)
    assert out[0, 0] / out[0, 1] == pytest.approx(0.6 / 0.4)
    np.testing.assert_allclose(out[1], X[1])  # rows without zeros are unchanged


def test_clr_rows_sum_to_zero_and_keep_frame_layout():
    df = pd.DataFrame([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5]], columns=["A", "B", "C"], index=[7, 9])
    clr = ClusteringModule._clr_transform(df, 0.005)
    assert list(clr.columns) == ["A", "B", "C"]
    assert list(clr.index) == [7, 9]
    assert np.all(np.isfinite(clr.to_numpy()))
    np.testing.assert_allclose(clr.sum(axis=1), 0.0, atol=1e-12)


def test_clr_drops_parts_that_are_always_zero():
    df = pd.DataFrame([[0.0, 0.6, 0.4], [0.0, 0.2, 0.8]], columns=["Li", "Nb", "O"])
    clr = ClusteringModule._clr_transform(df, 0.005)
    assert list(clr.columns) == ["Nb", "O"]
    np.testing.assert_allclose(clr, ClusteringModule._clr_transform(df[["Nb", "O"]], 0.005))


def test_clr_is_scale_invariant():
    df = pd.DataFrame([[0.2, 0.3, 0.5]], columns=["A", "B", "C"])
    np.testing.assert_allclose(
        ClusteringModule._clr_transform(df, 0.005), ClusteringModule._clr_transform(df * 100, 0.005)
    )


def test_get_clustering_features_respects_geometry():
    df = pd.DataFrame([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]], columns=["A", "B", "C"])
    euclid = SimpleNamespace(clustering_cfg=ClusteringConfig(geometry="euclidean"), verbose=False)
    aitch = SimpleNamespace(clustering_cfg=ClusteringConfig(geometry="aitchison"), verbose=False)
    assert ClusteringModule._get_clustering_features(euclid, df) is df
    np.testing.assert_allclose(
        ClusteringModule._get_clustering_features(aitch, df), ClusteringModule._clr_transform(df, 0.005)
    )


@pytest.mark.parametrize("bad_dl", [0, -0.1, 100, float("nan"), float("inf")])
def test_aitchison_params_rejects_invalid_detection_limit(bad_dl):
    with pytest.raises(ValueError):
        AitchisonParams(detection_limit_percent=bad_dl)


def test_detection_limit_floors_trace_values_and_preserves_major_ratios():
    # Column 2 is a trace element (median 0.3% < 2 x 0.5%)
    X = np.array([[0.60, 0.397, 0.003], [0.60, 0.40, 0.0], [0.5, 0.3, 0.2]])
    out = ClusteringModule._multiplicative_replacement(X, 0.005)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    # 0.3% and 0% are both below the 0.5% limit -> identical floored value
    assert out[0, 2] == pytest.approx(0.65 * 0.005)
    assert out[1, 2] == pytest.approx(0.65 * 0.005)
    assert out[0, 0] / out[0, 1] == pytest.approx(0.60 / 0.397)
    np.testing.assert_allclose(out[2], X[2])  # nothing below the limit -> unchanged


def test_detection_limit_keeps_low_values_of_major_elements():
    """A major element (median above 2 x limit) keeps its measured sub-limit values; only zeros are replaced."""
    X = np.array([[0.50, 0.20, 0.30], [0.50, 0.25, 0.25], [0.60, 0.003, 0.397], [0.60, 0.0, 0.40]])
    out = ClusteringModule._multiplicative_replacement(X, 0.005)
    assert out[2, 1] == pytest.approx(0.003)  # measured 0.3% of a major element is kept
    assert out[3, 1] == pytest.approx(0.65 * 0.005)  # exact zero -> 0.65 x limit


def test_detection_limit_removes_trace_element_log_ratio_gap():
    """0 vs 0.3% of a trace element must not create a large Aitchison distance."""
    df = pd.DataFrame([[0.17, 0.16, 0.667, 0.003], [0.17, 0.16, 0.67, 0.0]], columns=["Sr", "W", "O", "Se"])
    # Naive zero replacement (0 -> 1e-4) for comparison
    naive = np.log(np.where(df.to_numpy() > 0, df.to_numpy(), 1e-4))
    no_dl = naive - naive.mean(axis=1, keepdims=True)
    dl = ClusteringModule._clr_transform(df, 0.005).to_numpy()
    assert np.linalg.norm(no_dl[0] - no_dl[1]) > 2.5
    assert np.linalg.norm(dl[0] - dl[1]) < 0.01


# --- End-to-end DBSCAN ------------------------------------------------------
def _minor_element_phases(seed=0, n=25):
    """Two phases with near-identical majors that differ only in a minor-element level (0.1% vs 1%)."""
    rng = np.random.default_rng(seed)
    def phase(minor):
        majors = rng.normal(0.5, 0.005, size=(n, 1))
        m = np.abs(rng.normal(minor, minor * 0.1, size=(n, 1)))
        X = np.hstack([majors, 1 - majors, m])
        return X / X.sum(axis=1, keepdims=True)
    return pd.DataFrame(np.vstack([phase(0.001), phase(0.01)]), columns=["Fe", "Ni", "Cr"])


def _fake_analyzer(geometry, eps, min_samples=3):
    return SimpleNamespace(
        clustering_cfg=ClusteringConfig(
            method="dbscan", geometry=geometry, dbscan=DBSCANParams(eps=eps, min_samples=min_samples)
        ),
        verbose=False,
    )


def test_aitchison_dbscan_separates_minor_element_phases_euclidean_does_not():
    df = _minor_element_phases()
    truth = np.array([0] * 25 + [1] * 25)

    euclid = _fake_analyzer("euclidean", eps=0.02)
    _, _, k_e, _, _ = ClusteringModule._run_dbscan_clustering(euclid, df)
    assert k_e == 1

    aitch = _fake_analyzer("aitchison", eps=0.5)
    clr = ClusteringModule._get_clustering_features(aitch, df)
    labels, centroids, k, sil_score, wcss = ClusteringModule._run_dbscan_clustering(aitch, df, clustering_df=clr)
    assert k == 2
    # Clusters match the ground truth up to label permutation
    assert len({(t, l) for t, l in zip(truth, labels) if l != -1}) == 2
    assert 0.0 < sil_score <= 1.0

    # Centroids and WCSS are arithmetic means / sums of squares in fraction space
    X = df.to_numpy()
    for i in range(k):
        np.testing.assert_allclose(centroids[i], X[labels == i].mean(axis=0))
    expected_wcss = sum(float(np.sum((X[labels == i] - centroids[i]) ** 2)) for i in range(k))
    assert wcss == pytest.approx(expected_wcss)


def test_aitchison_dbscan_uses_clr_default_eps(monkeypatch):
    """With eps unset, DBSCAN in Aitchison geometry uses the CLR-scale default (0.3)."""
    import autoemx.core.composition_analysis.clustering as clustering_mod
    used = {}
    real_dbscan = clustering_mod.DBSCAN

    def spy_dbscan(**kwargs):
        used["eps"] = kwargs["eps"]
        return real_dbscan(**kwargs)

    monkeypatch.setattr(clustering_mod, "DBSCAN", spy_dbscan)
    df = _minor_element_phases()
    analyzer = SimpleNamespace(clustering_cfg=ClusteringConfig(method="dbscan", geometry="aitchison"), verbose=False)
    clr = ClusteringModule._get_clustering_features(analyzer, df)
    _, _, k, _, _ = ClusteringModule._run_dbscan_clustering(analyzer, df, clustering_df=clr, geometry="aitchison")
    assert used["eps"] == 0.3 and k == 2
    ClusteringModule._run_dbscan_clustering(analyzer, df, geometry="euclidean")
    assert used["eps"] == 0.05


# --- k-finding (hybrid) -----------------------------------------------------
def test_find_optimal_k_runs_single_cluster_check_on_fractions(monkeypatch):
    df = _minor_element_phases()
    clr = ClusteringModule._clr_transform(df, 0.005)
    seen = {}

    def fake_is_single(compositions_df, verbose=False):
        seen["single"] = compositions_df
        return False

    def fake_most_freq_k(compositions_df, max_k, method, verbose=False):
        seen["k_search"] = compositions_df
        return 2

    monkeypatch.setattr(ClusteringModule, "_is_single_cluster", staticmethod(fake_is_single))
    monkeypatch.setattr(ClusteringModule, "_get_most_freq_k", staticmethod(fake_most_freq_k))
    analyzer = SimpleNamespace(clustering_cfg=ClusteringConfig(geometry="aitchison"), verbose=False)

    k = ClusteringModule._find_optimal_k(analyzer, df, None, clustering_df=clr)

    assert k == 2
    assert seen["single"] is df
    assert seen["k_search"] is clr


# --- Automatic geometry selection -------------------------------------------
def _auto_analyzer():
    return SimpleNamespace(clustering_cfg=ClusteringConfig(geometry="auto"), verbose=False)


def test_clustering_config_accepts_auto_geometry():
    assert ClusteringConfig(geometry="AUTO").geometry == "auto"


def test_auto_geometry_picks_euclidean_for_two_elements():
    rng = np.random.default_rng(0)
    w = rng.normal(0.2, 0.01, size=30)
    df = pd.DataFrame({"Li": 0.0, "W": w, "O": 1 - w})  # Li undetectable -> only W and O present
    assert ClusteringModule._resolve_geometry(_auto_analyzer(), df) == "euclidean"


def test_auto_geometry_picks_euclidean_when_major_element_often_near_zero():
    rng = np.random.default_rng(1)
    main = rng.normal([0.15, 0.15, 0.70], 0.005, size=(40, 3))
    grains = np.column_stack([rng.uniform(0, 0.005, 10), rng.normal(0.33, 0.005, 10), rng.normal(0.67, 0.005, 10)])
    df = pd.DataFrame(np.vstack([main, grains]), columns=["Na", "Ti", "O"])  # Na < 1% in 20% of spectra
    n_el, frac = ClusteringModule._auto_geometry_indicators(df, 0.5, 1.0)
    assert n_el == 3 and frac == pytest.approx(0.2)
    assert ClusteringModule._resolve_geometry(_auto_analyzer(), df) == "euclidean"


def test_auto_geometry_picks_aitchison_for_ratio_differences():
    df = _minor_element_phases()  # Fe/Ni majors everywhere; Cr is trace -> not counted as near-zero major
    n_el, frac = ClusteringModule._auto_geometry_indicators(df, 0.5, 1.0)
    assert n_el == 3 and frac == 0.0
    assert ClusteringModule._resolve_geometry(_auto_analyzer(), df) == "aitchison"


def test_get_clustering_features_uses_resolved_auto_geometry():
    df = _minor_element_phases()
    np.testing.assert_allclose(
        ClusteringModule._get_clustering_features(_auto_analyzer(), df), ClusteringModule._clr_transform(df, 0.005)
    )


@pytest.mark.parametrize("field,bad", [("auto_near_zero_percent", 0), ("auto_near_zero_percent", 100),
                                       ("auto_max_near_zero_fraction", 0), ("auto_max_near_zero_fraction", 1.5)])
def test_aitchison_params_rejects_invalid_auto_thresholds(field, bad):
    with pytest.raises(ValueError):
        AitchisonParams(**{field: bad})
