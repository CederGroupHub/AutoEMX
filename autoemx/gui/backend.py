#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend of the AutoEMX sample-analysis GUI (no Dash code).

- Finds AutoEMX sample folders (folders containing ``ledger.json``).
- Describes every quantification/analysis parameter, with its value saved in the ledger.
- Runs ``batch_quantify_and_analyze`` / ``analyze_sample`` / single-spectrum fits in a
  separate process, capturing their log.
- Loads the results of any clustering analysis saved in a sample, for interactive plotting.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import signal
import sys
import tempfile
import threading
import time
import traceback
import types
import typing
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import autoemx.utils.constants as cnst
from autoemx.utils.helper import parse_xsp_spots_image_name
from autoemx.config.ledger_io import load_sample_ledger
from autoemx.config.ledger_schemas import (
    AitchisonParams,
    ClusterMergeParams,
    ClusteringConfig,
    DBSCANParams,
    MixtureParams,
)
from autoemx.config.runtime_configs import PlotConfig, QuantificationOptionsConfig

LEDGER_NAME = f"{cnst.LEDGER_FILENAME}{cnst.LEDGER_FILEEXT}"
_DEFAULT_UNDETECTABLE_ELS = ("H", "He", "Li")
_SKIPPED_DIRS = {"spectra", "__pycache__", ".git"}

QUANT_FLAG_MEANINGS: Dict[int, str] = {
    0: "OK",
    -1: "OK, not converged",
    1: "Acquisition error / no data",
    2: "Total counts too low",
    3: "Low-energy background too low",
    4: "Poor fit",
    5: "Analytical error > 50 w%",
    6: "Excessive X-ray absorption",
    7: "Substrate contamination",
    8: "Background under reference peak too low",
    9: "Fit interrupted (unknown)",
}


# =============================================================================
# Sample discovery and ledger summaries
# =============================================================================
@dataclass
class SampleRef:
    sample_id: str
    sample_dir: str
    label: str


def find_samples(root: str, max_depth: int = 5) -> List[SampleRef]:
    """Return every folder under ``root`` (included) that contains a ``ledger.json``."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")
    samples: List[SampleRef] = []
    root_depth = len(root_path.parts)
    for dirpath, dirnames, filenames in os.walk(root_path):
        depth = len(Path(dirpath).parts) - root_depth
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _SKIPPED_DIRS and not d.startswith(("analysis_quant", "."))
        )
        if depth >= max_depth:
            dirnames[:] = []
        if LEDGER_NAME in filenames:
            path = Path(dirpath)
            rel = path.relative_to(root_path)
            label = path.name if str(rel) == "." else str(rel)
            samples.append(SampleRef(sample_id=path.name, sample_dir=str(path), label=label))
    return sorted(samples, key=lambda s: s.label.lower())


@dataclass
class AnalysisRef:
    """One clustering analysis saved in the ledger, and the folder holding its outputs."""

    quant_id: int
    clust_index: int
    clust_id: int
    folder: str
    exists: bool
    is_active: bool
    config: ClusteringConfig
    result: Optional[Any]

    @property
    def key(self) -> str:
        return f"{self.quant_id}:{self.clust_index}"

    @property
    def label(self) -> str:
        cfg = self.config
        k = len(self.result.centroids) if self.result is not None else None
        parts = [f"quant{self.quant_id} / clust{self.clust_id}", cfg.method]
        if k is not None:
            parts.append(f"k={k}")
        parts.append(cfg.geometry)
        if cfg.ref_formulae:
            parts.append(", ".join(cfg.ref_formulae[:4]) + ("…" if len(cfg.ref_formulae) > 4 else ""))
        label = " · ".join(parts)
        if self.is_active:
            label += "  (active)"
        if not self.exists:
            label += "  [no output folder]"
        return label


@dataclass
class SampleInfo:
    sample_id: str
    sample_dir: str
    elements: List[str]
    substrate: List[str]
    n_spectra: int
    n_quantified: int
    quant_ids: List[int]
    active_quant: Optional[int]
    analyses: List[AnalysisRef]
    ledger: Any = field(repr=False)

    @property
    def active_analysis(self) -> Optional[AnalysisRef]:
        active = [a for a in self.analyses if a.is_active]
        return active[-1] if active else (self.analyses[-1] if self.analyses else None)


def _active_quant_config(ledger) -> Optional[Any]:
    if not ledger.quantifications:
        return None
    return next(
        (q for q in ledger.quantifications if q.quantification_id == ledger.active_quant),
        ledger.quantifications[-1],
    )


def undetectable_elements(ledger) -> List[str]:
    """Elements the sample's microscope cannot detect, from its calibration module."""
    import autoemx.calibrations as calibs

    try:
        calibs.load_microscope_calibrations(
            ledger.configs.microscope_cfg.ID, ledger.configs.measurement_cfg.mode
        )
        return list(getattr(calibs, "undetectable_els", _DEFAULT_UNDETECTABLE_ELS))
    except Exception:
        return list(_DEFAULT_UNDETECTABLE_ELS)


def load_sample_info(sample_dir: str) -> SampleInfo:
    """Load the ledger of a sample and summarise it."""
    ledger = load_sample_ledger(os.path.join(sample_dir, LEDGER_NAME))
    active_q = _active_quant_config(ledger)
    analyses: List[AnalysisRef] = []
    for q in ledger.quantifications:
        is_active_quant = active_q is not None and q.quantification_id == active_q.quantification_id
        for i, ca in enumerate(q.clustering_analyses):
            folder = os.path.join(
                sample_dir, f"analysis_quant{q.quantification_id}_clust{ca.config.clustering_id}"
            )
            analyses.append(
                AnalysisRef(
                    quant_id=q.quantification_id,
                    clust_index=i,
                    clust_id=ca.config.clustering_id,
                    folder=folder,
                    exists=os.path.isdir(folder),
                    is_active=is_active_quant and q.active_clustering_analysis_index == i,
                    config=ca.config,
                    result=ca.result,
                )
            )
    elements = list(active_q.sample_elements) if active_q and active_q.sample_elements else list(ledger.configs.sample_cfg.elements)
    substrate = list(active_q.substrate_elements) if active_q and active_q.substrate_elements else list(ledger.configs.sample_substrate_cfg.elements)
    n_quantified = 0
    if active_q is not None:
        for sp in ledger.spectra:
            if any(
                r.quantification_id == active_q.quantification_id and r.composition_atomic_fractions
                for r in sp.quantification_results
            ):
                n_quantified += 1
    return SampleInfo(
        sample_id=Path(sample_dir).name,
        sample_dir=str(sample_dir),
        elements=elements,
        substrate=substrate,
        n_spectra=len(ledger.spectra),
        n_quantified=n_quantified,
        quant_ids=[q.quantification_id for q in ledger.quantifications],
        active_quant=ledger.active_quant,
        analyses=analyses,
        ledger=ledger,
    )


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class ParamSpec:
    """One GUI control. ``key`` is '<section>.<name>'."""

    section: str
    name: str
    label: str
    kind: str  # bool, int, float, int_opt, float_opt, choice, text, formulae, elements, flags, pair
    default: Any = None
    choices: Optional[List[Any]] = None
    help: str = ""

    @property
    def key(self) -> str:
        return f"{self.section}.{self.name}"


SECTIONS: List[Tuple[str, str]] = [
    ("quant", "Quantification"),
    ("filter", "Spectra filtering"),
    ("clust", "Clustering"),
    ("dbscan", "DBSCAN"),
    ("aitchison", "Aitchison geometry"),
    ("merge", "Cluster merging"),
    ("mixture", "Mixture decomposition"),
    ("plot", "Plot output"),
]

# Sub-models whose fields are all exposed. New fields added to these models appear in the GUI automatically.
_SUBMODEL_SECTIONS: Dict[str, type] = {
    "dbscan": DBSCANParams,
    "aitchison": AitchisonParams,
    "merge": ClusterMergeParams,
    "mixture": MixtureParams,
}

_HELP: Dict[str, str] = {
    # Quantification
    "quant.els_sample": "Elements quantified in the sample. Changing them creates a new quantification run.",
    "quant.els_substrate": "Elements fitted but excluded from quantification (e.g. C, O, Al for carbon tape).",
    "quant.min_bckgrnd_cnts": "Minimum background counts under the reference peaks. Spectra below are flagged (quant_flag 8).",
    "quant.min_total_counts_fraction": "Minimum total counts, as a fraction of the target acquisition counts. Spectra below are flagged (quant_flag 2). 0 disables the check.",
    "quant.spectrum_lims": "Lower and upper channel indices of the fitted spectral range.",
    "quant.fit_tolerance": "Fit tolerance used when re-fitting single spectra in the viewer (batch quantification keeps the saved value).",
    "quant.use_instrument_background": "Use the instrument background saved with each spectrum instead of computing it during the fit.",
    "quant.use_project_specific_std_dict": "Load the P/B standards file from the results folder instead of the default calibration file.",
    "quant.interrupt_fits_bad_spectra": "Stop fitting spectra as soon as they are found to give a poor quantification. Much faster.",
    "quant.force_requantification": "Quantify all spectra again even if a run with the same settings exists.",
    "quant.num_CPU_cores": "CPU cores used for fitting. Empty = half of the available cores.",
    "quant.is_known_precursor_mixture": "Sample is a mixture of known powders: characterizes their extent of intermixing.",
    # Filtering
    "filter.max_analytical_error_percent": "Spectra whose analytical error exceeds this value (w%) are excluded from clustering. Empty disables the check.",
    "filter.quant_flags_accepted": "Quantification flags kept for clustering.",
    # Clustering
    "clust.method": "Clustering algorithm.",
    "clust.geometry": "'aitchison' clusters log-ratios (CLR) of compositions; 'auto' chooses per sample.",
    "clust.features": "Cluster atomic (at_fr) or mass (w_fr) fractions.",
    "clust.k_forced": "Number of clusters (k-means). Empty = determined automatically.",
    "clust.k_finding_method": "Method used to find the number of clusters automatically.",
    "clust.max_k": "Maximum number of clusters tested when k is found automatically.",
    "clust.auto_merge_clusters": "Merge k-means clusters that are pieces of one continuous population (e.g. a mixture line).",
    "clust.ref_formulae": "Candidate phases, one formula per line or comma-separated (e.g. MgO, Al2O3, MgAl2O4).",
    "clust.do_matrix_decomposition": "Decompose clusters into mixtures of candidate phases. Slow with many candidates.",
    # DBSCAN
    "dbscan.eps": "Neighbourhood radius. Empty = 0.05 (Euclidean) or 0.3 (Aitchison).",
    "dbscan.min_samples": "Minimum points in a neighbourhood to form a dense region.",
    "dbscan.metric": "Distance metric.",
    # Aitchison
    "aitchison.detection_limit_percent": "Zeros and sub-detection values of trace elements are replaced with 0.65 × this limit before the log-ratio transform.",
    "aitchison.auto_near_zero_percent": "'auto' geometry: threshold below which a major element counts as near zero.",
    "aitchison.auto_max_near_zero_fraction": "'auto' geometry: use Euclidean if at least this fraction of spectra has a near-zero major element.",
    # Merge
    "merge.dip_alpha": "Dip-test p-value above which the projection of two clusters is unimodal.",
    "merge.max_gap_ratio": "Maximum empty gap between two clusters, relative to the spread of the larger one.",
    "merge.dbscan_min_samples": "min_samples of the DBSCAN connectivity check.",
    "merge.dbscan_eps_factor": "eps of the connectivity check, as a multiple of the median k-NN distance.",
    # Mixture
    "mixture.max_n_phases": "Maximum number of candidate phases combined in a mixture (2 = binary only).",
    "mixture.max_recon_error": "Reconstruction error below which a mixture explains a cluster.",
    "mixture.max_recon_error_binary": "Binary mixtures below this error are listed (for inspection).",
    "mixture.recon_error_alpha": "Exponent of the reconstruction-error metric.",
    "mixture.conf_sigma": "Width of the error → confidence mapping.",
    "mixture.single_phase_max_rms_dist": "Clusters tighter than this RMS distance that match a candidate are single-phase.",
    "mixture.single_phase_min_ref_conf": "Minimum candidate confidence for a single-phase cluster.",
    "mixture.nmf_min_mixture_conf": "If no mixture reaches this confidence, a free NMF decomposition is also run.",
    "mixture.equivalent_recon_error_tol": "Mixtures whose errors differ less than this are equivalent.",
    "mixture.max_reported_mixtures": "Maximum mixtures listed per cluster in Clusters.csv.",
    "mixture.report_within_conf_ratio": "Mixtures within this ratio of the best confidence are always listed.",
    "mixture.min_reported_conf_ratio": "Mixtures below this ratio of the best confidence are not listed.",
    "mixture.collapse_equivalent_mixtures": "Hide mixtures spanning the same mixing line/plane as a better one.",
    "mixture.equivalent_span_tol": "Tolerance (fractions) for equivalent mixing lines/planes.",
    # Plot
    "plot.els_to_plot": "Elements forced onto the axes of the saved clustering plot (2 or 3).",
    "plot.els_excluded_clust_plot": "Elements excluded from the saved clustering plot.",
    "plot.show_unused_comps_clust": "Show discarded compositions in the saved clustering plot.",
    "plot.show_legend_clustering": "Show the legend in the saved clustering plot.",
    "plot.plot_best_mixture": "Save a plot of the best mixture of each cluster.",
}

_CHOICES: Dict[str, List[Any]] = {
    "clust.method": list(ClusteringConfig.ALLOWED_METHODS),
    "clust.geometry": list(ClusteringConfig.ALLOWED_GEOMETRIES),
    "clust.features": [cnst.AT_FR_CL_FEAT, cnst.W_FR_CL_FEAT],
    "clust.k_finding_method": ["silhouette", "calinski_harabasz", "elbow"],
    "dbscan.metric": ["euclidean", "manhattan", "chebyshev", "cosine"],
}


def _kind_from_annotation(annotation: Any) -> str:
    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    is_optional = origin in (typing.Union, types.UnionType) and len(args) == 1
    base = args[0] if is_optional else annotation
    if base is bool:
        return "bool"
    if base is int:
        return "int_opt" if is_optional else "int"
    if base is float:
        return "float_opt" if is_optional else "float"
    return "text"


def _model_specs(section: str, model: type, names: Optional[List[str]] = None) -> List[ParamSpec]:
    specs = []
    for name, info in model.model_fields.items():
        if names is not None and name not in names:
            continue
        key = f"{section}.{name}"
        kind = "choice" if key in _CHOICES else _kind_from_annotation(info.annotation)
        default = info.get_default(call_default_factory=True)
        specs.append(
            ParamSpec(
                section=section,
                name=name,
                label=name.replace("_", " "),
                kind=kind,
                default=default,
                choices=_CHOICES.get(key),
                help=_HELP.get(key, info.description or ""),
            )
        )
    if names is not None:
        specs.sort(key=lambda s: names.index(s.name))
    return specs


def build_param_specs() -> List[ParamSpec]:
    """All GUI parameters, in display order."""
    qdef = QuantificationOptionsConfig()
    specs: List[ParamSpec] = [
        ParamSpec("quant", "els_sample", "sample elements", "text", ""),
        ParamSpec("quant", "els_substrate", "substrate elements", "text", ""),
        ParamSpec("quant", "min_bckgrnd_cnts", "min background counts", "float_opt",
                  ClusteringConfig().min_bckgrnd_cnts),
        ParamSpec("quant", "min_total_counts_fraction", "min total counts fraction", "float",
                  qdef.min_total_counts_fraction),
        ParamSpec("quant", "spectrum_lims", "spectrum limits (channels)", "pair", list(qdef.spectrum_lims)),
        ParamSpec("quant", "fit_tolerance", "fit tolerance (viewer)", "float", qdef.fit_tolerance),
        ParamSpec("quant", "use_instrument_background", "use instrument background", "bool",
                  qdef.use_instrument_background),
        ParamSpec("quant", "use_project_specific_std_dict", "project-specific standards", "bool",
                  qdef.use_project_specific_std_dict),
        ParamSpec("quant", "interrupt_fits_bad_spectra", "interrupt fits of bad spectra", "bool", True),
        ParamSpec("quant", "force_requantification", "force requantification", "bool", False),
        ParamSpec("quant", "num_CPU_cores", "CPU cores", "int_opt", None),
        ParamSpec("quant", "is_known_precursor_mixture", "known precursor mixture", "bool", False),
        ParamSpec("filter", "max_analytical_error_percent", "max analytical error (w%)", "float_opt", 5.0),
        ParamSpec("filter", "quant_flags_accepted", "accepted quant flags", "flags", [0, -1],
                  choices=list(QUANT_FLAG_MEANINGS)),
    ]
    specs += _model_specs(
        "clust", ClusteringConfig,
        ["method", "geometry", "features", "k_forced", "k_finding_method", "max_k",
         "auto_merge_clusters", "ref_formulae", "do_matrix_decomposition"],
    )
    for section, model in _SUBMODEL_SECTIONS.items():
        specs += _model_specs(section, model)
    specs += _model_specs(
        "plot", PlotConfig,
        ["els_to_plot", "els_excluded_clust_plot", "show_unused_comps_clust",
         "show_legend_clustering", "plot_best_mixture"],
    )
    for s in specs:
        if s.key in ("clust.ref_formulae",):
            s.kind = "formulae"
        elif s.key in ("plot.els_to_plot", "plot.els_excluded_clust_plot"):
            s.kind = "elements"
        if not s.help:
            s.help = _HELP.get(s.key, "")
    return specs


PARAM_SPECS: List[ParamSpec] = build_param_specs()
SPECS_BY_KEY: Dict[str, ParamSpec] = {s.key: s for s in PARAM_SPECS}


def sample_param_values(info: SampleInfo, analysis: Optional[AnalysisRef] = None) -> Dict[str, Any]:
    """Parameter values saved in the ledger (from ``analysis``, or the active analysis)."""
    ledger = info.ledger
    values = {s.key: s.default for s in PARAM_SPECS}
    active_q = _active_quant_config(ledger)
    if analysis is not None:
        active_q = next((q for q in ledger.quantifications if q.quantification_id == analysis.quant_id), active_q)

    values["quant.els_sample"] = ", ".join(info.elements)
    values["quant.els_substrate"] = ", ".join(info.substrate)
    if active_q is not None:
        if active_q.sample_elements:
            values["quant.els_sample"] = ", ".join(active_q.sample_elements)
        values["quant.els_substrate"] = ", ".join(active_q.substrate_elements)
        try:
            opts = QuantificationOptionsConfig(**active_q.options)
            values["quant.min_total_counts_fraction"] = opts.min_total_counts_fraction
            values["quant.spectrum_lims"] = [int(round(v)) for v in opts.spectrum_lims]
            values["quant.fit_tolerance"] = opts.fit_tolerance
            values["quant.use_instrument_background"] = opts.use_instrument_background
            values["quant.use_project_specific_std_dict"] = opts.use_project_specific_std_dict
        except Exception:
            pass
    powder = ledger.configs.measurement_cfg.powder_meas_cfg
    if powder is not None:
        values["quant.is_known_precursor_mixture"] = bool(powder.is_known_powder_mixture_meas)

    if analysis is None:
        analysis = info.active_analysis
    cfg = analysis.config if analysis is not None else ClusteringConfig()
    values["quant.min_bckgrnd_cnts"] = cfg.min_bckgrnd_cnts
    values["filter.max_analytical_error_percent"] = cfg.max_analytical_error_percent
    values["filter.quant_flags_accepted"] = list(cfg.quant_flags_accepted)
    for name in ("method", "geometry", "features", "k_forced", "max_k",
                 "auto_merge_clusters", "ref_formulae", "do_matrix_decomposition"):
        values[f"clust.{name}"] = getattr(cfg, name)
    values["clust.k_finding_method"] = (
        cfg.k_finding_method if cfg.k_finding_method != "forced"
        else ClusteringConfig.model_fields["k_finding_method"].default
    )
    for section, attr in (("dbscan", "dbscan"), ("aitchison", "aitchison"),
                          ("merge", "cluster_merge"), ("mixture", "mixture")):
        for name, value in getattr(cfg, attr).model_dump().items():
            values[f"{section}.{name}"] = value
    plot_cfg = ledger.configs.plot_cfg
    for name in ("els_to_plot", "els_excluded_clust_plot", "show_unused_comps_clust",
                 "show_legend_clustering", "plot_best_mixture"):
        values[f"plot.{name}"] = getattr(plot_cfg, name)
    return values


def _elements_list(text: Any) -> List[str]:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [str(t) for t in text if str(t).strip()]
    from autoemx.web.pipeline import parse_elements

    return parse_elements(str(text))


def _formulae_list(text: Any) -> List[str]:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        items = text
    else:
        items = str(text).replace("\n", ",").replace(";", ",").split(",")
    return [t.strip() for t in items if str(t).strip()]


def coerce_values(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert GUI values to typed values. Raises ``ValueError`` listing every invalid field."""
    out: Dict[str, Any] = {}
    errors: List[str] = []
    for key, spec in SPECS_BY_KEY.items():
        value = raw.get(key, spec.default)
        try:
            if spec.kind == "bool":
                out[key] = bool(value) if not isinstance(value, list) else bool(value)
            elif spec.kind in ("int", "int_opt"):
                if value in (None, ""):
                    if spec.kind == "int":
                        raise ValueError("a value is required")
                    out[key] = None
                else:
                    f = float(value)
                    if f != int(f):
                        raise ValueError("must be an integer")
                    out[key] = int(f)
            elif spec.kind in ("float", "float_opt"):
                if value in (None, ""):
                    if spec.kind == "float":
                        raise ValueError("a value is required")
                    out[key] = None
                else:
                    out[key] = float(value)
            elif spec.kind == "pair":
                vals = value if isinstance(value, (list, tuple)) else str(value).replace(";", ",").split(",")
                vals = [int(float(v)) for v in vals if str(v).strip() != ""]
                if len(vals) != 2 or vals[0] >= vals[1]:
                    raise ValueError("needs two increasing integers")
                out[key] = vals
            elif spec.kind == "flags":
                out[key] = sorted({int(v) for v in (value or [])})
            elif spec.kind == "formulae":
                out[key] = _formulae_list(value)
            elif spec.kind == "elements":
                out[key] = _elements_list(value)
            elif spec.key in ("quant.els_sample", "quant.els_substrate"):
                out[key] = _elements_list(value)
            else:
                out[key] = value
        except Exception as exc:
            errors.append(f"{dict(SECTIONS)[spec.section]} › {spec.label}: {exc}")
    errors += _validate_models(out)
    if errors:
        raise ValueError("; ".join(errors))
    return out


def _section_dict(values: Dict[str, Any], section: str) -> Dict[str, Any]:
    prefix = section + "."
    return {k[len(prefix):]: v for k, v in values.items() if k.startswith(prefix)}


def _validate_models(values: Dict[str, Any]) -> List[str]:
    errors = []
    for section, model in _SUBMODEL_SECTIONS.items():
        try:
            section_values = _section_dict(values, section)
            if len(section_values) < len(model.model_fields):
                continue  # a field failed coercion, already reported
            model.model_validate(section_values)
        except Exception as exc:
            errors.append(f"{dict(SECTIONS)[section]}: {_pydantic_msg(exc)}")
    from pymatgen.core import Composition, Element

    for formula in values.get("clust.ref_formulae", []):
        try:
            valid = all(isinstance(el, Element) for el in Composition(formula).elements)
        except Exception:
            valid = False
        if not valid:
            errors.append(f"Invalid candidate formula '{formula}'")
    if "quant.els_sample" in values and not values["quant.els_sample"]:
        errors.append("At least one sample element is required")
    if values.get("filter.quant_flags_accepted") == []:
        errors.append("Accept at least one quant flag")
    return errors


def _pydantic_msg(exc: Exception) -> str:
    errs = getattr(exc, "errors", None)
    if callable(errs):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errs())
    return str(exc)


def analysis_kwargs(values: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for ``analyze_sample`` from coerced GUI values."""
    k_forced = values["clust.k_forced"]
    kwargs = dict(
        ref_formulae=values["clust.ref_formulae"],
        clustering_features=values["clust.features"],
        clustering_method=values["clust.method"],
        clustering_geometry=values["clust.geometry"],
        dbscan_params=_section_dict(values, "dbscan"),
        aitchison_params=_section_dict(values, "aitchison"),
        k_forced=k_forced if k_forced is not None else False,
        k_finding_method=None if k_forced is not None else values["clust.k_finding_method"],
        max_k=values["clust.max_k"],
        auto_merge_clusters=values["clust.auto_merge_clusters"],
        cluster_merge_params=_section_dict(values, "merge"),
        do_matrix_decomposition=values["clust.do_matrix_decomposition"],
        mixture_params=_section_dict(values, "mixture"),
        max_analytical_error_percent=values["filter.max_analytical_error_percent"],
        quant_flags_accepted=values["filter.quant_flags_accepted"],
        min_bckgrnd_cnts=values["quant.min_bckgrnd_cnts"],
        els_to_plot=values["plot.els_to_plot"],
        els_excluded_clust_plot=values["plot.els_excluded_clust_plot"],
        show_unused_compositions_cluster_plot=values["plot.show_unused_comps_clust"],
        show_legend_clustering=values["plot.show_legend_clustering"],
        plot_best_mixture=values["plot.plot_best_mixture"],
        show_plots=False,
        plot_custom_plots=False,
    )
    return kwargs


def quantification_kwargs(values: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword arguments for ``batch_quantify_and_analyze`` from coerced GUI values."""
    return dict(
        els_sample=values["quant.els_sample"],
        els_substrate=values["quant.els_substrate"],
        quantification_method="PB",
        min_bckgrnd_cnts=values["quant.min_bckgrnd_cnts"],
        min_total_counts_fraction=values["quant.min_total_counts_fraction"],
        spectrum_lims=tuple(values["quant.spectrum_lims"]),
        use_instrument_background=values["quant.use_instrument_background"],
        use_project_specific_std_dict=values["quant.use_project_specific_std_dict"],
        interrupt_fits_bad_spectra=values["quant.interrupt_fits_bad_spectra"],
        force_requantification=values["quant.force_requantification"],
        num_CPU_cores=values["quant.num_CPU_cores"],
        is_known_precursor_mixture=values["quant.is_known_precursor_mixture"],
        run_analysis=False,
    )


# =============================================================================
# Job runner (separate process, log captured to file)
# =============================================================================
def _job_entry(kind: str, sample_dir: str, payload: Dict[str, Any], log_path: str, result_path: str) -> None:
    """Child-process entry point."""
    if hasattr(os, "setpgrp"):
        os.setpgrp()  # own process group, so cancelling also stops worker pools
    os.environ["MPLBACKEND"] = "Agg"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    import matplotlib

    matplotlib.use("Agg")
    result: Dict[str, Any] = {"ok": False}
    try:
        result = _JOB_FUNCS[kind](sample_dir, payload)
        result.setdefault("ok", True)
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.stdout.flush()
        tmp = result_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(result, fh, default=_json_default)
        os.replace(tmp, result_path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _run_quant_and_analysis(sample_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if payload.get("quantify"):
        from autoemx.runners.batch_quantify_and_analyze import batch_quantify_and_analyze

        print("=" * 60 + "\nQUANTIFICATION\n" + "=" * 60, flush=True)
        res = batch_quantify_and_analyze(
            sample_IDs=[Path(sample_dir).name],
            results_path=str(Path(sample_dir).parent),
            **payload["quant_kwargs"],
        )
        if not res:
            raise RuntimeError("Quantification failed. See the log for details.")
        out["quantified"] = True
    if payload.get("analyse", True):
        from autoemx.runners.analyze_sample import analyze_sample

        print("=" * 60 + "\nANALYSIS\n" + "=" * 60, flush=True)
        analyzer = analyze_sample(
            sample_ID=Path(sample_dir).name,
            results_path=str(Path(sample_dir).parent),
            **payload["analysis_kwargs"],
        )
        if analyzer is None:
            raise RuntimeError("Analysis failed. See the log for details.")
        out["analysis_dir"] = getattr(analyzer, "analysis_dir", None)
        out["n_clusters"] = (getattr(analyzer, "clustering_info", None) or {}).get("n_clusters")
        if out["n_clusters"] is None:
            out["warning"] = "Analysis did not produce clusters (too few valid spectra?). See the log."
    return out


def _run_fit(sample_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from autoemx.runners.fit_and_quantify_spectrum_from_ledger import fit_and_quantify_spectrum_from_ledger
    from autoemx.web.pipeline import _peak_labels, _plot_series

    info = load_sample_info(sample_dir)
    sample_cfg = info.ledger.configs.sample_cfg
    quantifier = fit_and_quantify_spectrum_from_ledger(
        sample_ID=info.sample_id,
        spectrum_ID=int(payload["spectrum_id"]),
        els_sample=info.elements,
        els_substrate=info.substrate,
        results_path=str(Path(sample_dir).parent),
        quantify_plot=True,
        plot_signal=False,
        fit_tol=float(payload.get("fit_tol", 1e-4)),
        is_particle=bool(sample_cfg.is_surface_rough),
        use_instrument_background=bool(payload.get("use_instrument_background", False)),
        print_results=False,
        quant_verbose=True,
        fitting_verbose=False,
        ledger=info.ledger,
    )
    if quantifier is None or getattr(quantifier, "fit_result", None) is None:
        raise RuntimeError("Fit failed. See the log for details.")
    energy, counts, fitted, background = _plot_series(quantifier)
    quant = getattr(quantifier, "quant_result", None) or {}
    try:
        bckgrnd_cnts = [[line, e, v] for line, (e, v) in quantifier.get_bckgrnd_cnts_ref_lines().items()]
    except Exception:
        traceback.print_exc()
        bckgrnd_cnts = []
    return {
        "spectrum_id": payload["spectrum_id"],
        "energy": energy,
        "counts": counts,
        "fit": fitted,
        "background": background,
        "peak_labels": _peak_labels(quantifier),
        "bckgrnd_cnts": bckgrnd_cnts,  # [el_line, energy, background counts] under each reference line
        "comp_at": dict(quant.get(cnst.COMP_AT_FR_KEY) or {}),
        "comp_w": dict(quant.get(cnst.COMP_W_FR_KEY) or {}),
        "analytical_error": quant.get(cnst.AN_ER_KEY),
        "r_squared": quant.get(cnst.R_SQ_KEY),
        "redchi_sq": quant.get(cnst.REDCHI_SQ_KEY),
    }


_JOB_FUNCS = {"analysis": _run_quant_and_analysis, "fit": _run_fit}


@dataclass
class Job:
    job_id: str
    kind: str
    sample_dir: str
    description: str
    process: Any
    log_path: str
    result_path: str
    started: float
    cancelled: bool = False

    def is_running(self) -> bool:
        return self.process.is_alive()

    def result(self) -> Optional[Dict[str, Any]]:
        if self.is_running() or not os.path.exists(self.result_path):
            if not self.is_running() and not os.path.exists(self.result_path):
                msg = "Cancelled." if self.cancelled else f"Process exited unexpectedly (code {self.process.exitcode})."
                return {"ok": False, "error": msg}
            return None
        with open(self.result_path, encoding="utf-8") as fh:
            return json.load(fh)

    def log_tail(self, max_chars: int = 20000) -> str:
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_chars))
                text = fh.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        return text if size <= max_chars else "…\n" + text.split("\n", 1)[-1]

    def elapsed(self) -> float:
        return time.time() - self.started


class JobManager:
    """Runs one job at a time per kind, each in its own process."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._tmpdir = tempfile.mkdtemp(prefix="autoemx_gui_")
        self._ctx = mp.get_context("spawn")

    def start(self, kind: str, sample_dir: str, payload: Dict[str, Any], description: str) -> Job:
        with self._lock:
            for job in self._jobs.values():
                if job.kind == kind and job.is_running():
                    raise RuntimeError(f"A {kind} job is already running: {job.description}")
            job_id = uuid.uuid4().hex[:10]
            log_path = os.path.join(self._tmpdir, f"{job_id}.log")
            result_path = os.path.join(self._tmpdir, f"{job_id}.json")
            open(log_path, "w").close()
            proc = self._ctx.Process(
                target=_job_entry,
                args=(kind, sample_dir, payload, log_path, result_path),
                daemon=False,
            )
            proc.start()
            job = Job(job_id, kind, sample_dir, description, proc, log_path, result_path, time.time())
            self._jobs[job_id] = job
            return job

    def get(self, job_id: Optional[str]) -> Optional[Job]:
        return self._jobs.get(job_id) if job_id else None

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or not job.is_running():
            return
        job.cancelled = True
        try:
            if hasattr(os, "killpg"):
                os.killpg(job.process.pid, signal.SIGTERM)
            else:
                job.process.terminate()
        except ProcessLookupError:
            pass
        job.process.join(timeout=5)
        if job.process.is_alive():
            job.process.kill()

    def shutdown(self) -> None:
        for job_id in list(self._jobs):
            self.cancel(job_id)


# =============================================================================
# Analysis results
# =============================================================================
@dataclass
class AnalysisData:
    """Everything needed to plot one clustering analysis."""

    sample_id: str
    folder: str
    quant_id: Optional[int]
    elements: List[str]
    detectable: List[str]
    features: str  # at_fr or w_fr
    comps: pd.DataFrame  # one row per spectrum, columns: spectrum, particle, cluster, status, <el> (fractions), ...
    centroids: np.ndarray  # (k, n_elements), fractions, in ``elements`` order
    stdevs: np.ndarray
    n_points: List[int]
    ref_formulae: List[str]
    ref_comps: pd.DataFrame  # index: formula, columns: elements (fractions)
    mixtures: List[List[Dict[str, Any]]]
    candidates: List[List[Tuple[str, float]]]  # per cluster: (candidate phase, confidence)
    clusters_table: Optional[pd.DataFrame]
    config: ClusteringConfig
    summary: Dict[str, Any]
    images: List[str]
    config_summary: str

    @property
    def unit(self) -> str:
        return "w%" if self.features == cnst.W_FR_CL_FEAT else "at%"


def reference_compositions(
    formulae: List[str], elements: List[str], features: str, undetectable: List[str]
) -> pd.DataFrame:
    """Candidate-phase compositions, computed as in the analyser (``_initialise_ref_phases``)."""
    from pymatgen.core import Composition

    from autoemx.utils import composition_as_weight_dict

    detectable = [el for el in elements if el not in undetectable]
    rows, names, seen = [], [], set()
    for formula in formulae:
        try:
            comp = Composition(formula)
        except Exception:
            continue
        if comp.reduced_formula in seen:
            continue
        seen.add(comp.reduced_formula)
        w_fr = composition_as_weight_dict(comp)
        if not any(el in w_fr for el in detectable):
            continue
        if features == cnst.W_FR_CL_FEAT:
            phase = {el: w_fr.get(el, 0.0) for el in detectable}
        else:
            det_w = {el: w for el, w in w_fr.items() if el in detectable}
            phase = comp.from_weight_dict(det_w).fractional_composition.as_dict()
        rows.append(phase)
        names.append(formula)
    df = pd.DataFrame(rows, columns=elements, index=names).fillna(0.0)
    return df.astype(float)


def _spectrum_key(spectrum_id: Any, index: int) -> str:
    sid = spectrum_id if spectrum_id is not None else index
    try:
        return str(int(float(sid)))
    except (TypeError, ValueError):
        return str(sid)


def _pixel(details: Any, axis: int) -> Optional[int]:
    coords = details.spot_coordinates if details is not None else None
    if coords is None or coords.pixel_coordinates is None:
        return None
    return int(coords.pixel_coordinates[axis])


def spectra_table(info: SampleInfo, quant_id: Optional[int], elements: List[str], features: str) -> pd.DataFrame:
    """One row per ledger spectrum with its composition for quantification ``quant_id``."""
    rows = []
    for idx, sp in enumerate(info.ledger.spectra):
        rec = next(
            (r for r in reversed(sp.quantification_results) if r.quantification_id == quant_id), None
        )
        details = sp.acquisition_details
        at = dict(rec.composition_atomic_fractions or {}) if rec else {}
        w = dict(rec.composition_weight_fractions or {}) if rec else {}
        fit = rec.fit_result if rec else None
        row: Dict[str, Any] = {
            "spectrum": _spectrum_key(sp.spectrum_id, idx),
            "particle": details.particle_id if details else None,
            "frame": details.frame_id if details else None,
            "total_counts": sp.total_counts,
            # Pixel position of the spot in the SEM image of its particle/frame
            "px": _pixel(details, 0),
            "py": _pixel(details, 1),
            "quant_flag": rec.quant_flag if rec else None,
            "an_err": rec.analytical_error * 100 if rec and rec.analytical_error is not None else None,
            "r_squared": fit.r_squared if fit else None,
            "redchi_sq": fit.reduced_chi_squared if fit else None,
            "comment": rec.comment if rec else None,
        }
        main = w if features == cnst.W_FR_CL_FEAT else at
        for el in elements:
            row[el] = main.get(el, 0.0) if main else np.nan
        for el in elements:
            row[f"{el} at%"] = at[el] * 100 if el in at else np.nan
        for el in elements:
            row[f"{el} w%"] = w[el] * 100 if el in w else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_labels(folder: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(folder, "Compositions.csv")
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path)
    if cnst.SP_ID_DF_KEY not in raw or cnst.CL_ID_DF_KEY not in raw:
        return None
    return {
        _spectrum_key(sid, i): cl
        for i, (sid, cl) in enumerate(zip(raw[cnst.SP_ID_DF_KEY], raw[cnst.CL_ID_DF_KEY]))
    }


def load_analysis(info: SampleInfo, analysis: Optional[AnalysisRef]) -> AnalysisData:
    """Load one clustering analysis (or only the quantified compositions, if ``analysis`` is None)."""
    quant_id = analysis.quant_id if analysis is not None else info.active_quant
    quant = next((q for q in info.ledger.quantifications if q.quantification_id == quant_id), None)
    elements = list(dict.fromkeys(quant.sample_elements if quant and quant.sample_elements else info.elements))
    cfg = analysis.config if analysis is not None else ClusteringConfig()
    features = cfg.features
    undetectable = undetectable_elements(info.ledger)
    folder = analysis.folder if analysis is not None else ""

    comps = spectra_table(info, quant_id, elements, features)
    labels = _cluster_labels(folder) if folder else None
    comps["cluster"] = comps["spectrum"].map(labels) if labels is not None else np.nan
    quantified = comps[elements].notna().any(axis=1)
    if labels is None:
        comps["status"] = np.where(quantified, "not analysed", "not quantified")
    else:
        cl = comps["cluster"]
        comps["status"] = np.where(
            ~quantified, "not quantified",
            np.where(cl.isna(), "discarded", np.where(cl.fillna(-1) >= 0, "clustered", "noise")),
        )

    result = analysis.result if analysis is not None else None
    if result is not None:
        # Ledger centroids are stored in the clustering features (at_fr or w_fr), in ``elements`` order.
        cent_src, std_src = result.centroids, result.els_std_dev_per_cluster
        centroids = np.atleast_2d(np.asarray(cent_src, dtype=float))
        stdevs = np.atleast_2d(np.asarray(std_src, dtype=float))
        n_points = list(result.n_points_per_cluster)
        mixtures = [list(m or []) for m in result.clusters_assigned_mixtures]
        candidates = [_candidate_matches(row) for row in (result.refs_assigned_rows or [])]
        summary = {
            "n_clusters": len(result.centroids),
            "wcss": result.wcss,
            "silhouette": result.sil_score,
            "n_total": result.tot_n_points,
        }
    else:
        centroids = np.zeros((0, len(elements)))
        stdevs = np.zeros((0, len(elements)))
        n_points, mixtures, candidates = [], [], []
        summary = {"n_clusters": 0}
    if centroids.size == 0:
        centroids = np.zeros((0, len(elements)))
        stdevs = np.zeros((0, len(elements)))
    counts = comps["status"].value_counts().to_dict()
    summary.update({
        "n_spectra": len(comps),
        "n_clustered": int(counts.get("clustered", 0)),
        "n_noise": int(counts.get("noise", 0)),
        "n_discarded": int(counts.get("discarded", 0)),
        "n_not_quantified": int(counts.get("not quantified", 0)),
        "has_labels": labels is not None,
    })

    clusters_table = None
    clusters_path = os.path.join(folder, "Clusters.csv")
    if folder and os.path.exists(clusters_path):
        try:
            clusters_table = pd.read_csv(clusters_path)
        except Exception:
            clusters_table = None
    images = sorted(str(p) for p in Path(folder).glob("*.png")) if folder else []
    config_summary = ""
    summary_path = os.path.join(folder, "Analysis_config_summary.txt")
    if folder and os.path.exists(summary_path):
        config_summary = Path(summary_path).read_text(encoding="utf-8", errors="replace")

    return AnalysisData(
        sample_id=info.sample_id,
        folder=folder,
        elements=elements,
        detectable=[el for el in elements if el not in undetectable],
        features=features,
        comps=comps,
        centroids=centroids,
        stdevs=stdevs,
        n_points=n_points,
        ref_formulae=list(cfg.ref_formulae),
        quant_id=quant_id,
        ref_comps=reference_compositions(list(cfg.ref_formulae), elements, features, undetectable),
        mixtures=mixtures,
        candidates=candidates,
        clusters_table=clusters_table,
        config=cfg,
        summary=summary,
        images=images,
        config_summary=config_summary,
    )


def _candidate_matches(row: Dict[str, Any]) -> List[Tuple[str, float]]:
    """(candidate, confidence) pairs from a ``refs_assigned_rows`` entry ({'Cnd1': ..., 'CS_cnd1': ...})."""
    out = []
    j = 1
    while f"Cnd{j}" in row:
        name, conf = row.get(f"Cnd{j}"), row.get(f"CS_cnd{j}")
        if name:
            out.append((str(name), float(conf) if conf is not None else float("nan")))
        j += 1
    return out


def find_analysis(info: SampleInfo, key: Optional[str]) -> Optional[AnalysisRef]:
    if key:
        for a in info.analyses:
            if a.key == key:
                return a
    return info.active_analysis


# =============================================================================
# Raw spectra
# =============================================================================
def load_raw_spectrum(info: SampleInfo, spectrum_id: Any) -> Dict[str, Any]:
    """Counts and energy axis of one spectrum, read from its file (no fit)."""
    ledger = info.ledger
    entry = None
    for idx, sp in enumerate(ledger.spectra):
        sid = sp.spectrum_id if sp.spectrum_id is not None else str(idx)
        try:
            match = int(float(sid)) == int(float(spectrum_id))
        except (TypeError, ValueError):
            match = str(sid) == str(spectrum_id)
        if match:
            entry = sp
            break
    if entry is None or not entry.spectrum_relpath:
        raise FileNotFoundError(f"Spectrum {spectrum_id} not found in the ledger.")
    counts = np.asarray(
        ledger._load_counts_from_pointer_file(Path(info.sample_dir) / entry.spectrum_relpath), dtype=float
    )
    mcfg = ledger.configs.microscope_cfg
    energy = mcfg.energy_zero + mcfg.bin_width * np.arange(len(counts))
    details = entry.acquisition_details
    return {
        "energy": energy,
        "counts": counts,
        "total_counts": entry.total_counts,
        "live_time": entry.live_acquisition_time,
        "particle": details.particle_id if details else None,
        "frame": details.frame_id if details else None,
        "path": str(Path(info.sample_dir) / entry.spectrum_relpath),
    }


# =============================================================================
# SEM images
# =============================================================================
SEM_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
# Whole frames, with the detected particles circled
_FRAME_RE = re.compile(r"_fr_?(?P<frame>.+?)_particles$")


@dataclass
class SemImage:
    """An SEM image saved during acquisition."""

    path: str
    name: str
    kind: str  # "spots" (spectrum spots of a particle or frame), "frame" (whole frame) or "other"
    particle: Optional[int] = None
    frame: Optional[str] = None

    @property
    def label(self) -> str:
        if self.kind == "spots" and self.particle is not None:
            return f"Particle {self.particle}" + (f" · frame {self.frame}" if self.frame not in (None, "None") else "")
        if self.kind in ("spots", "frame") and self.frame is not None:
            return f"Frame {self.frame}"
        return self.name


def find_sem_images(sample_dir: str) -> List[SemImage]:
    """SEM images of a sample: particle/frame images with spectrum spots first, then the others."""
    images: List[SemImage] = []
    folders = [Path(sample_dir) / cnst.IMAGES_DIR, Path(sample_dir)]
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SEM_IMAGE_EXTENSIONS or path.stem.endswith("_raw"):
                continue
            parsed = parse_xsp_spots_image_name(path.stem)
            if parsed is not None:
                images.append(SemImage(str(path), path.name, "spots", parsed[0], parsed[1]))
            else:
                frame_match = _FRAME_RE.search(path.stem)
                if frame_match:
                    images.append(SemImage(str(path), path.name, "frame", frame=frame_match.group("frame")))
                else:
                    images.append(SemImage(str(path), path.name, "other"))

    def natural(text: Any) -> List[Any]:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]

    def order(im: SemImage):
        rank = {"spots": 0, "frame": 1, "other": 2}[im.kind]
        return (rank, im.particle if im.particle is not None else -1, natural(im.frame), natural(im.name))

    return sorted(images, key=order)


def sem_image_for_spectrum(images: List[SemImage], particle: Any, frame: Any) -> Optional[SemImage]:
    """Image showing the spot of a spectrum: the image of its particle, else of its frame."""
    spots = [im for im in images if im.kind == "spots"]
    try:
        par = int(float(particle))
    except (TypeError, ValueError):
        par = None
    if par is not None:
        matches = [im for im in spots if im.particle == par]
        if matches:
            exact = [im for im in matches if frame is not None and str(im.frame) == str(frame)]
            return (exact or matches)[0]
        return None
    if frame not in (None, "", "None"):
        matches = [im for im in spots if im.particle is None and str(im.frame) == str(frame)]
        if matches:
            return matches[0]
    return None


def spectra_on_image(comps: pd.DataFrame, image: SemImage) -> pd.DataFrame:
    """Rows of ``comps`` whose spots are in ``image`` (with pixel coordinates)."""
    if image.kind != "spots":
        return comps.iloc[0:0]
    rows = comps[comps["px"].notna() & comps["py"].notna()]
    if image.particle is not None:
        par = pd.to_numeric(rows["particle"], errors="coerce")
        return rows[par == image.particle]
    return rows[rows["frame"].astype(str) == str(image.frame)]


def read_image_png(path: str, max_size: Optional[int] = None) -> Tuple[bytes, Tuple[int, int]]:
    """PNG bytes of an image (first page of TIFFs), optionally downscaled, and its original size."""
    import io

    from PIL import Image

    with Image.open(path) as im:
        size = im.size
        im = im.convert("RGB")
        if max_size:
            im.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=False)
    return buf.getvalue(), size
