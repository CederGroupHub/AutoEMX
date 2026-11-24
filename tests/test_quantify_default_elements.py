#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_quantify_and_analyze without els_sample quantifies the elements of the active quantification.

These are the elements shown for the sample (e.g. in the Quantification tab of the GUI, which passes no elements
when they are unchanged), not those of the ledger's base sample config. No spectra are quantified: the analyzer
built by the runner is intercepted.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from ci_paths import INPUTS_DIR, WULFENITE_MINI_ID

from autoemx.core.composition_analysis import EMXSp_Composition_Analyzer
from autoemx.runners.batch_quantify_and_analyze import batch_quantify_and_analyze


class _Stop(Exception):
    pass


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    sample_dir = Path(shutil.copytree(INPUTS_DIR / WULFENITE_MINI_ID, tmp_path / WULFENITE_MINI_ID))
    ledger_path = sample_dir / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    # Active quantification with an element added after acquisition (base sample config: Pb, Mo, O)
    assert ledger["configs"]["sample_cfg"]["elements"] == ["Pb", "Mo", "O"]
    ledger["quantifications"][0]["sample_elements"] = ["Pb", "Mo", "O", "Cu"]
    ledger_path.write_text(json.dumps(ledger))
    return tmp_path


def _elements_quantified(monkeypatch, results_dir: Path, **kwargs) -> list:
    seen = []

    def fake_from_ledger(ledger, cfgs, **_kw):
        seen.append(list(cfgs.sample_cfg.elements))
        raise _Stop

    monkeypatch.setattr(EMXSp_Composition_Analyzer, "from_ledger", staticmethod(fake_from_ledger))
    try:
        batch_quantify_and_analyze(sample_IDs=[WULFENITE_MINI_ID], results_path=str(results_dir),
                                   run_analysis=False, **kwargs)
    except _Stop:
        pass
    assert len(seen) == 1
    return seen[0]


def test_default_elements_are_those_of_active_quantification(monkeypatch, results_dir):
    assert _elements_quantified(monkeypatch, results_dir) == ["Pb", "Mo", "O", "Cu"]


def test_elements_appended_to_active_quantification(monkeypatch, results_dir):
    assert _elements_quantified(monkeypatch, results_dir, els_sample=["", "S"]) == ["Pb", "Mo", "O", "Cu", "S"]


def test_given_elements_replace(monkeypatch, results_dir):
    assert _elements_quantified(monkeypatch, results_dir, els_sample=["Pb", "Mo"]) == ["Pb", "Mo"]
