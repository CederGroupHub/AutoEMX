#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report of an acquisition started from the GUI by an external program (see ``open_acquisition``).

The report of the run ``run_id`` is kept in ``<system temp folder>/autoemx_runs/<run_id>.json`` while the
run goes on. It is written when the run starts, after each sample, and when the run ends (also when it
is stopped or fails). It is a dict::

    {
        "run_id": str,
        "status": "running" | "finished" | "stopped" | "failed",
        "error": None or why the run stopped or failed,
        "started": ISO date, "finished": ISO date or None,
        "results_folder": str,
        "requested_samples": IDs of the samples given by the external program, or None,
        "samples": [{
            "ID": str,
            "path": folder of the sample,
            "status": "waiting" | "running" | "done" | "failed" | "stopped" | "not_run",
            "error": None or why the sample failed or was not completed,
            "n_spectra": spectra in the sample folder,
            "n_new_spectra": spectra acquired during this run,
            "n_spectra_before": spectra in the sample folder before this run,
        }, ...],
    }

External programs read it with ``acquisition_report`` (current state) or ``wait_for_acquisition`` (waits for
the end of the run, then deletes the file). This module does not import Dash, so that they can read reports
cheaply.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import autoemx.utils.constants as cnst

REPORTS_DIR = Path(tempfile.gettempdir()) / "autoemx_runs"
LEDGER_NAME = f"{cnst.LEDGER_FILENAME}{cnst.LEDGER_FILEEXT}"
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,100}")


def clean_run_id(run_id: Any) -> str:
    """The run ID, checked: 1 to 100 letters, digits, '-' or '_' (it names the report file)."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("'run_id' must be 1 to 100 letters, digits, '-' or '_'")
    return run_id


def report_path(run_id: str) -> Path:
    return REPORTS_DIR / f"{clean_run_id(run_id)}.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def count_spectra(sample_dir: str) -> int:
    """Number of spectra in the ledger of a sample folder (0 if it has none yet)."""
    try:
        with open(os.path.join(sample_dir, LEDGER_NAME), encoding="utf-8") as fh:
            return len(json.load(fh).get("spectra") or [])
    except (OSError, ValueError, AttributeError):
        return 0


def new_report(run_id: str, sample_ids: List[str], results_dir: str,
               requested: Optional[List[str]] = None) -> Dict[str, Any]:
    """Report of a run that is starting (its first sample is running)."""
    samples = []
    for sid in sample_ids:
        path = os.path.join(results_dir, sid)
        n = count_spectra(path)
        samples.append({"ID": sid, "path": path, "status": "waiting", "error": None,
                        "n_spectra": n, "n_new_spectra": 0, "n_spectra_before": n})
    if samples:
        samples[0]["status"] = "running"
    return {"run_id": clean_run_id(run_id), "status": "running", "error": None, "started": _now(),
            "finished": None, "results_folder": results_dir, "requested_samples": requested, "samples": samples}


def _update_counts(sample: Dict[str, Any]) -> None:
    sample["n_spectra"] = count_spectra(sample["path"])
    sample["n_new_spectra"] = max(0, sample["n_spectra"] - sample["n_spectra_before"])


def sample_done(report: Dict[str, Any], sample_id: str, error: Optional[BaseException]) -> None:
    """Record the end of a sample (``error``: None if it succeeded); the next sample is then running."""
    samples = report["samples"]
    i = next((k for k, s in enumerate(samples) if s["ID"] == sample_id and s["status"] == "running"), None)
    if i is None:
        return
    samples[i]["status"] = "failed" if error is not None else "done"
    samples[i]["error"] = f"{type(error).__name__}: {error}" if error is not None else None
    _update_counts(samples[i])
    if i + 1 < len(samples) and samples[i + 1]["status"] == "waiting":
        samples[i + 1]["status"] = "running"


def finish_report(report: Dict[str, Any], status: str, error: Optional[str] = None) -> None:
    """Record the end of the run: 'finished', 'stopped' or 'failed' (``error``: why)."""
    report["status"], report["error"], report["finished"] = status, error, _now()
    for smp in report["samples"]:
        if smp["status"] == "running":
            # Interrupted while being acquired (with 'finished', the batch ended before reporting it)
            smp["status"] = "stopped" if status == "stopped" else "failed"
            smp["error"] = error or "The run ended before this sample was completed"
        elif smp["status"] == "waiting":
            smp["status"] = "not_run"
            smp["error"] = error or "The run ended before this sample"
        _update_counts(smp)


def write_report(report: Dict[str, Any]) -> Path:
    """Write the report (atomically, so that a reader never sees a partly written file)."""
    path = report_path(report["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    os.replace(tmp, path)
    return path


def read_report(run_id: str) -> Optional[Dict[str, Any]]:
    """The report of the run, or None if there is none (the run was not started yet)."""
    try:
        with open(report_path(run_id), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


FINAL_STATUSES = ("finished", "stopped", "failed")


def acquisition_report(run_id: str) -> Optional[Dict[str, Any]]:
    """Current report of the run *run_id* (see ``open_acquisition``), or None if the user has not started it yet.

    Its ``status`` is 'running' until the run ends, then 'finished', 'stopped' or 'failed'.
    """
    return read_report(run_id)


def wait_for_acquisition(run_id: str, timeout: Optional[float] = None, poll_interval: float = 2.0,
                         delete: bool = True) -> Dict[str, Any]:
    """
    Wait for the end of the run *run_id* (see ``open_acquisition``) and return its final report. Also waits for
    the user to start the run. Meant to be called in a thread of the calling program.

    With *delete*, the report file is deleted once read; if the user starts another acquisition from the
    same page, call this again to get its report. Raises ``TimeoutError`` after *timeout* seconds (None: no limit).
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        report = read_report(run_id)
        if report is not None and report.get("status") in FINAL_STATUSES:
            if delete:
                try:
                    report_path(run_id).unlink()
                except FileNotFoundError:
                    pass
            return report
        if deadline is not None and time.monotonic() >= deadline:
            state = "not started" if report is None else "still running"
            raise TimeoutError(f"AutoEMX run {run_id} {state} after {timeout} s")
        time.sleep(poll_interval if deadline is None else max(0.0, min(poll_interval, deadline - time.monotonic())))


def follow_job(job: Any, report: Dict[str, Any]) -> None:
    """Wait for the acquisition job to end, then make its report final (run in a thread of the GUI server).

    The job process updates the report after each sample, and makes it final when the run finishes or
    fails. It is completed here when the job was stopped or its process died.
    """
    job.process.join()
    latest = read_report(report["run_id"])
    if latest is None or latest.get("started") != report["started"]:
        latest = report  # the job process could not write it
    elif latest["status"] != "running":
        return  # made final by the job process
    result = job.result() or {}
    if job.cancelled:
        finish_report(latest, "stopped", "Acquisition stopped (Stop button, or AutoEMX GUI closed)")
    elif result.get("ok"):
        finish_report(latest, "finished")
    else:
        finish_report(latest, "failed", result.get("error") or "The acquisition process ended unexpectedly")
    write_report(latest)
