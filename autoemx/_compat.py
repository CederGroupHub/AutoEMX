"""Compatibility helpers for package-rename transition."""

from __future__ import annotations

import glob
import os
import shlex
import site
import sys
import warnings


def _candidate_site_packages() -> list[str]:
    paths: set[str] = set()
    for getter in (site.getsitepackages,):
        try:
            for p in getter():
                if p:
                    paths.add(os.path.abspath(p))
        except Exception:
            continue

    try:
        user_site = site.getusersitepackages()
        if user_site:
            paths.add(os.path.abspath(user_site))
    except Exception:
        pass

    return sorted(paths)


def _find_stale_autoemxsp_artifacts() -> list[str]:
    stale: list[str] = []
    for site_pkg in _candidate_site_packages():
        ns_dir = os.path.join(site_pkg, "autoemxsp")
        init_file = os.path.join(ns_dir, "__init__.py")
        if os.path.isdir(ns_dir) and not os.path.exists(init_file):
            stale.append(ns_dir)

        stale.extend(glob.glob(os.path.join(site_pkg, "__editable__.autoemxsp-*.pth")))
        stale.extend(glob.glob(os.path.join(site_pkg, "__editable___autoemxsp_*_finder.py")))

    # De-duplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for path in stale:
        norm = os.path.abspath(path)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _build_cleanup_command(paths: list[str]) -> str:
    exe = shlex.quote(sys.executable)
    quoted_paths = ", ".join(repr(p) for p in paths)
    py_cleanup = (
        "import os,shutil; "
        f"paths=[{quoted_paths}]; "
        "[shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.remove(p) if os.path.exists(p) else None) for p in paths]"
    )
    return (
        f"{exe} -m pip uninstall -y autoemxsp && "
        f"{exe} -c \"{py_cleanup}\" && "
        f"{exe} -m pip install -U autoemx"
    )


def warn_if_stale_autoemxsp_install() -> None:
    """Warn users when stale legacy package artifacts may shadow the shim."""
    stale_paths = _find_stale_autoemxsp_artifacts()
    if not stale_paths:
        return

    fix_cmd = _build_cleanup_command(stale_paths)
    warnings.warn(
        "Detected stale 'autoemxsp' installation artifacts that can break legacy imports "
        "(e.g. 'autoemxsp.runners'). Run this one-time fix:\n"
        f"  {fix_cmd}",
        FutureWarning,
        stacklevel=2,
    )
