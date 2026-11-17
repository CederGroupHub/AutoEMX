#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings and samples of an acquisition script, for the "Load script…" button of the Acquisition tab.

Reads scripts like ``Run_Acquisition.py``, ``Collect_Particle_Statistics.py`` or those saved by "Save as script…":
the keyword arguments of their call to ``batch_acquire_and_analyze`` or ``collect_particle_statistics``, given
directly or through variables defined earlier in the script. The script is never run: values are read from its
source code, and only literals, earlier variables, arithmetic and ``dict(...)`` are understood. Settings set
otherwise (e.g. ``os.path.dirname(__file__)``) are reported as not loaded.

Samples commented out in the ``samples`` list (``# {'ID': ..., ...},``) are loaded too, not ticked for acquisition.
"""

from __future__ import annotations

import ast
import io
import math
import operator
import tokenize
import uuid
from typing import Any, Dict, List, Optional, Tuple

RUNNERS = {"batch_acquire_and_analyze": "eds", "collect_particle_statistics": "sizes"}

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
_UNRESOLVED = object()  # variables whose value could not be read


class _Unresolved(Exception):
    pass


def _eval(node: ast.AST, env: Dict[str, Any]) -> Any:
    """Value of an expression made of literals, known variables, arithmetic and dict(...) (never runs code)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_eval(e, env) for e in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else set(items) if isinstance(node, ast.Set) else items
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise _Unresolved
        return {_eval(k, env): _eval(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        value = env.get(node.id, _UNRESOLVED)
        if value is _UNRESOLVED:
            raise _Unresolved
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left, env), _eval(node.right, env)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (left, right)):
            raise _Unresolved
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        if any(kw.arg is None for kw in node.keywords):
            raise _Unresolved
        return {kw.arg: _eval(kw.value, env) for kw in node.keywords}
    raise _Unresolved


def _runner_call(tree: ast.Module) -> Optional[ast.Call]:
    """The (last) call to an acquisition runner in the script."""
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ((isinstance(n.func, ast.Name) and n.func.id in RUNNERS)
                  or (isinstance(n.func, ast.Attribute) and n.func.attr in RUNNERS))]
    return max(calls, key=lambda n: n.lineno) if calls else None


def _top_level_env(tree: ast.Module, before_line: int) -> Tuple[Dict[str, Any], Dict[str, ast.AST]]:
    """Variables assigned at the top level of the script before ``before_line``: their values (``_UNRESOLVED`` if
    they cannot be read) and the expression last assigned to each."""
    env: Dict[str, Any] = {}
    nodes: Dict[str, ast.AST] = {}
    for stmt in tree.body:
        if stmt.lineno >= before_line:
            break
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name, value = stmt.target.id, stmt.value
        else:
            continue
        try:
            env[name] = _eval(value, env)
        except (_Unresolved, ArithmeticError, TypeError, ValueError):
            env[name] = _UNRESOLVED
        nodes[name] = value
    return env, nodes


def _commented_samples(source: str, first_line: int, last_line: int) -> List[Tuple[int, Dict[str, Any]]]:
    """Samples commented out between two lines of the script (e.g. inside the ``samples`` list), with their line:
    comments holding a dict with an 'ID', possibly spread over consecutive comment lines."""
    comments: List[Tuple[int, str]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and first_line <= tok.start[0] <= last_line:
                comments.append((tok.start[0], tok.string.lstrip("#").strip()))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    out = []
    i = 0
    while i < len(comments):
        line, text = comments[i]
        if not text.startswith(("{", "dict(")):
            i += 1
            continue
        # A dict may continue on the next comment lines
        for j in range(i, min(i + 10, len(comments))):
            if j > i and comments[j][0] != comments[j - 1][0] + 1:
                break
            candidate = " ".join(t for _, t in comments[i:j + 1]).rstrip().rstrip(",")
            try:
                value = _eval(ast.parse(candidate, mode="eval").body, {})
            except (SyntaxError, _Unresolved, ArithmeticError, TypeError, ValueError):
                continue
            if isinstance(value, dict) and value.get("ID") is not None:
                out.append((line, value))
                i = j
            break
        i += 1
    return out


def _row(sample: Dict[str, Any], acquire: bool) -> Dict[str, Any]:
    """Row of the samples list of the Acquisition tab from a sample of a script."""
    def text(value):
        return ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value or "")

    pos = sample.get("pos")
    x, y = (pos[0], pos[1]) if isinstance(pos, (list, tuple)) and len(pos) == 2 else (None, None)
    return {"uid": uuid.uuid4().hex[:8], "on": ["on"] if acquire else [], "ID": str(sample.get("ID") or ""),
            "els": text(sample.get("els")), "slot": None, "x": x, "y": y, "cnd": text(sample.get("cnd"))}


def _diameter(area: Any) -> float:
    """Particle diameter (µm) of a particle area (µm²), as in the GUI (areas of circles)."""
    return round(math.sqrt(4 * float(area) / math.pi), 6)


def _form_values(kwargs: Dict[str, Any], mode: str, form_keys: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Values of the acquisition form (keys 'section.name') from the arguments of the runner, and the arguments
    that have no field in the form."""
    keys = set(form_keys)
    values: Dict[str, Any] = {}
    unused: List[str] = []

    def put(key, value):
        if key in keys:
            values[key] = value
            return True
        return False

    def put_powder(cfg, size_only):
        for name, value in (cfg or {}).items():
            if name in ("min_area_par", "max_area_par") and size_only:
                put(f"asize.{name[:3]}_par_diameter_um", _diameter(value))
            elif name == "size_stats_ci_tolerance":
                put("asize.size_stats_ci_tolerance", value * 100 if value is not None else None)
            elif name in ("par_selection_mode", "max_n_par_per_frame") and size_only:
                continue  # set by the GUI for particle size statistics
            elif not (put(f"asize.{name}", value) if size_only or name.startswith(("size_stats", "min_par_pixels",
                                                                                   "include_eds"))
                      else put(f"apowder.{name}", value)):
                unused.append(f"{name} (particle settings)")

    quantify = kwargs.get("quantify_spectra")
    for name, value in kwargs.items():
        if name in ("samples", "results_dir", "development_mode", "verbose", "on_sample_done"):
            continue
        if name == "microscope_ID":
            put("amicro.microscope_ID", value)
        elif name == "els_substrate":
            put("asample.els_substrate", ", ".join(value) if isinstance(value, (list, tuple)) else value)
        elif name in ("min_n_spectra", "max_n_spectra"):
            if quantify:
                put(f"aacq.{name}", value)
            elif name == "max_n_spectra":
                put("aacq.n_spectra", value)  # without quantification, the number of spectra collected
        elif name == "max_XSp_acquisition_time":
            # Empty field: the default time for the target counts (as in Run_Acquisition.py)
            target = kwargs.get("target_Xsp_counts")
            is_default = isinstance(target, (int, float)) and value is not None and abs(value - target / 10000 * 5) < 1e-9
            put("aacq.max_XSp_acquisition_time", None if is_default else value)
        elif name == "powder_meas_cfg_kwargs":
            put_powder(value, size_only=mode == "sizes")
        elif name == "size_stats_powder_meas_cfg_kwargs":
            put_powder(value, size_only=True)
        elif name == "bulk_meas_cfg_kwargs":
            for k, v in (value or {}).items():
                if not put(f"abulk.{k}", v):
                    unused.append(f"{k} (bulk settings)")
        elif name in ("size_stats_n_par_target", "n_par_target"):
            put("asize.n_par_target", value)
        elif not any(put(f"{section}.{name}", value) for section in ("asample", "aacq", "aimg", "aquant")):
            unused.append(name)
    return values, unused


def load_acquisition_script(source: str, form_keys: List[str]) -> Dict[str, Any]:
    """
    Settings and samples of an acquisition script.

    Returns ``{"mode", "values", "rows", "folder", "not_loaded"}``: the measurement mode ('eds', 'eds+sizes' or
    'sizes'), the values of the acquisition form (keys 'section.name', among ``form_keys``) set by the script,
    the rows of the samples list (``on``: ticked for acquisition, i.e. not commented out), the results folder
    (None if not given as plain text), and the settings that could not be loaded. Raises ``ValueError`` if the
    script cannot be read or does not run an acquisition.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"not a valid Python script (line {exc.lineno}: {exc.msg})") from None
    call = _runner_call(tree)
    if call is None:
        raise ValueError("no call to " + " or ".join(RUNNERS) + " found in the script")
    runner = call.func.id if isinstance(call.func, ast.Name) else call.func.attr
    env, nodes = _top_level_env(tree, call.lineno)

    kwargs: Dict[str, Any] = {}
    not_loaded: List[str] = []
    samples_node: Optional[ast.AST] = None
    for kw in call.keywords:
        if kw.arg is None:
            not_loaded.append("**" + (ast.unparse(kw.value) if hasattr(ast, "unparse") else "kwargs"))
            continue
        if kw.arg == "samples":
            samples_node = nodes.get(kw.value.id) if isinstance(kw.value, ast.Name) else kw.value
        try:
            kwargs[kw.arg] = _eval(kw.value, env)
        except (_Unresolved, ArithmeticError, TypeError, ValueError):
            if kw.arg not in ("development_mode", "verbose", "on_sample_done"):
                not_loaded.append(kw.arg)

    # Samples: those of the list (ticked), and those commented out in it (not ticked), in the order of the script
    samples = kwargs.get("samples")
    if not isinstance(samples, list) or not isinstance(samples_node, ast.List):
        raise ValueError("the samples must be given as a list of dicts ('samples = [{...}, ...]')")
    entries = [(e.lineno, s, True) for e, s in zip(samples_node.elts, samples) if isinstance(s, dict)]
    entries += [(line, s, False) for line, s in
                _commented_samples(source, samples_node.lineno, samples_node.end_lineno or samples_node.lineno)]
    rows = [_row(s, acquire) for _, s, acquire in sorted(entries, key=lambda e: e[0])]

    mode = RUNNERS[runner]
    if mode == "eds" and kwargs.get("size_stats_n_par_target"):
        mode = "eds+sizes"
    values, unused = _form_values(kwargs, mode, form_keys)
    folder = kwargs.get("results_dir")
    return {"mode": mode, "values": values, "rows": rows,
            "folder": folder if isinstance(folder, str) and folder else None,
            "not_loaded": not_loaded + unused}
