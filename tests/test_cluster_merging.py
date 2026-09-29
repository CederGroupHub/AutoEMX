#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the dip test and for merging k-means clusters that form one continuous population."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autoemx.config.ledger_schemas import ClusteringAnalysis, ClusteringConfig, ClusterMergeParams
from autoemx.core.composition_analysis.clustering import ClusteringModule
from autoemx.utils.unimodality import dip_statistic, dip_test


# --- Dip test ------------------------------------------------------------------
def test_dip_statistic_bounds():
    rng = np.random.default_rng(0)
    for n in (5, 30, 200):
        d = dip_statistic(rng.normal(size=n))
        assert 1 / (2 * n) - 1e-12 <= d <= 0.25


def test_dip_statistic_known_values():
    # Two equal point masses: the largest possible dip for a two-point sample
    assert dip_statistic([0, 0, 0, 1, 1, 1]) == pytest.approx(0.25)
    # Evenly spaced values are unimodal (uniform): dip = 1 / (2n)
    assert dip_statistic(np.arange(10)) == pytest.approx(1 / 20)


def test_dip_test_separates_unimodal_and_bimodal():
    rng = np.random.default_rng(1)
    _, p_uniform = dip_test(rng.random(150))
    _, p_bimodal = dip_test(np.r_[rng.normal(size=75), rng.normal(size=75) + 6])
    assert p_uniform > 0.2
    assert p_bimodal < 0.01


def test_dip_test_small_samples():
    assert dip_test([1.0, 2.0, 3.0])[1] == 1.0
    assert dip_statistic([2.0, 2.0, 2.0]) == 0.0


# --- Config -----------------------------------------------------------------------
def test_auto_merge_defaults_true_for_new_configs():
    cfg = ClusteringConfig()
    assert cfg.auto_merge_clusters is True
    assert cfg.cluster_merge == ClusterMergeParams()


def test_saved_configs_without_field_do_not_merge():
    legacy = ClusteringAnalysis.model_validate({"config": {"method": "kmeans", "features": "at_fr"}})
    assert legacy.config.auto_merge_clusters is False
    saved = ClusteringAnalysis.model_validate({"config": {"method": "kmeans", "auto_merge_clusters": True}})
    assert saved.config.auto_merge_clusters is True


def test_fingerprint_unchanged_when_not_merging():
    payload = ClusteringConfig(auto_merge_clusters=False).fingerprint_payload()
    assert "auto_merge_clusters" not in payload and "cluster_merge" not in payload
    assert ClusteringConfig().fingerprint() != ClusteringConfig(auto_merge_clusters=False).fingerprint()


def test_legacy_config_loader_disables_merging(tmp_path):
    from autoemx.utils.legacy.ledger_bootstrap import _load_legacy_clustering_cfg
    (tmp_path / "Comp_analysis_configs.json").write_text(json.dumps({"clustering_cfg": {"method": "kmeans", "k": 2}}))
    cfg = _load_legacy_clustering_cfg(str(tmp_path))
    assert cfg is not None and cfg.auto_merge_clusters is False


@pytest.mark.parametrize("kwargs", [
    {"dip_alpha": 0}, {"dip_alpha": 1}, {"max_gap_ratio": 0}, {"dbscan_eps_factor": -1}, {"dbscan_min_samples": 1},
])
def test_cluster_merge_params_validation(kwargs):
    with pytest.raises(ValueError):
        ClusterMergeParams(**kwargs)


# --- Merging ----------------------------------------------------------------------
def _stub():
    return SimpleNamespace(clustering_cfg=ClusteringConfig(), verbose=False)


def test_mixture_line_cut_by_kmeans_is_merged():
    """Compositions spread evenly along a line, split into three pieces, form one cluster."""
    rng = np.random.default_rng(2)
    t = rng.random(120)
    X = np.c_[t, 1 - t] + rng.normal(0, 0.01, (120, 2))
    labels = np.digitize(t, [1 / 3, 2 / 3])
    merged, k = ClusteringModule._merge_connected_clusters(_stub(), pd.DataFrame(X), labels)
    assert k == 1 and set(merged) == {0}


def test_separated_groups_are_kept():
    rng = np.random.default_rng(3)
    X = np.r_[rng.normal(0, 0.01, (60, 2)), rng.normal(0, 0.01, (40, 2)) + [0.2, -0.2]]
    labels = np.r_[np.zeros(60, int), np.ones(40, int)]
    _, k = ClusteringModule._merge_connected_clusters(_stub(), pd.DataFrame(X), labels)
    assert k == 2


def test_small_separate_group_is_kept():
    """A few compositions clear of the main group (e.g. a minor phase) stay a separate cluster."""
    rng = np.random.default_rng(4)
    X = np.r_[rng.normal(0, 0.01, (50, 2)), rng.normal(0, 0.003, (3, 2)) + [0.06, 0.0]]
    labels = np.r_[np.zeros(50, int), np.ones(3, int)]
    _, k = ClusteringModule._merge_connected_clusters(_stub(), pd.DataFrame(X), labels)
    assert k == 2


def test_merged_labels_are_renumbered():
    rng = np.random.default_rng(5)
    t = rng.random(90)
    X = np.c_[t, 1 - t] + rng.normal(0, 0.01, (90, 2))
    X = np.r_[X, rng.normal(0, 0.01, (40, 2)) + [3.0, 3.0]]
    labels = np.r_[np.digitize(t, [0.5]) * 2, np.ones(40, int)]  # labels 0, 2 (line) and 1 (separate group)
    merged, k = ClusteringModule._merge_connected_clusters(_stub(), pd.DataFrame(X), labels)
    assert k == 2 and sorted(set(merged)) == [0, 1]
