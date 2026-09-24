#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit UI: upload EMSA spectra, fit/quantify, download PNG and TXT."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

# Repo root on sys.path so the app runs on Streamlit Cloud without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from autoemx.web.exports import (
    figure_to_png_bytes,
    fitted_spectrum_figure,
    format_batch_composition_txt,
    format_composition_txt,
)
from autoemx.web.pipeline import (
    QUANT_BEAM_KV,
    QUANTIFIABLE_ELEMENTS,
    SUPPORTED_UPLOAD_EXTENSIONS,
    SpectrumFitResult,
    beam_energy_supports_quantification,
    fit_uploaded_spectrum,
    parse_elements,
    validate_elements_quantifiable,
)
from autoemx.web.reader_report import (
    GITHUB_NEW_ISSUE_URL,
    REPORT_EMAIL,
    SpectrumReadError,
    mailto_reader_report_url,
)

_DEMO_FILE_CAP = 2
_ACCEPT = sorted(SUPPORTED_UPLOAD_EXTENSIONS)
_LICENSE_CONTACT = "IPO@lbl.gov"

# (row, col) positions in the standard 18-column periodic table layout.
# Lanthanides and actinides are placed in rows 9 and 10 (with a gap after row 7).
_PT_POSITIONS: dict[str, tuple[int, int]] = {
    "H": (1, 1), "He": (1, 18),
    "Li": (2, 1), "Be": (2, 2), "B": (2, 13), "C": (2, 14), "N": (2, 15), "O": (2, 16), "F": (2, 17), "Ne": (2, 18),
    "Na": (3, 1), "Mg": (3, 2), "Al": (3, 13), "Si": (3, 14), "P": (3, 15), "S": (3, 16), "Cl": (3, 17), "Ar": (3, 18),
    "K": (4, 1), "Ca": (4, 2), "Sc": (4, 3), "Ti": (4, 4), "V": (4, 5), "Cr": (4, 6), "Mn": (4, 7), "Fe": (4, 8), "Co": (4, 9), "Ni": (4, 10), "Cu": (4, 11), "Zn": (4, 12), "Ga": (4, 13), "Ge": (4, 14), "As": (4, 15), "Se": (4, 16), "Br": (4, 17), "Kr": (4, 18),
    "Rb": (5, 1), "Sr": (5, 2), "Y": (5, 3), "Zr": (5, 4), "Nb": (5, 5), "Mo": (5, 6), "Tc": (5, 7), "Ru": (5, 8), "Rh": (5, 9), "Pd": (5, 10), "Ag": (5, 11), "Cd": (5, 12), "In": (5, 13), "Sn": (5, 14), "Sb": (5, 15), "Te": (5, 16), "I": (5, 17), "Xe": (5, 18),
    "Cs": (6, 1), "Ba": (6, 2), "La": (6, 3), "Hf": (6, 4), "Ta": (6, 5), "W": (6, 6), "Re": (6, 7), "Os": (6, 8), "Ir": (6, 9), "Pt": (6, 10), "Au": (6, 11), "Hg": (6, 12), "Tl": (6, 13), "Pb": (6, 14), "Bi": (6, 15), "Po": (6, 16), "At": (6, 17), "Rn": (6, 18),
    "Fr": (7, 1), "Ra": (7, 2), "Ac": (7, 3), "Rf": (7, 4), "Db": (7, 5), "Sg": (7, 6), "Bh": (7, 7), "Hs": (7, 8), "Mt": (7, 9), "Ds": (7, 10), "Rg": (7, 11), "Cn": (7, 12), "Nh": (7, 13), "Fl": (7, 14), "Mc": (7, 15), "Lv": (7, 16), "Ts": (7, 17), "Og": (7, 18),
    # Lanthanides (row 9, gap after row 7)
    "Ce": (9, 4), "Pr": (9, 5), "Nd": (9, 6), "Pm": (9, 7), "Sm": (9, 8), "Eu": (9, 9), "Gd": (9, 10), "Tb": (9, 11), "Dy": (9, 12), "Ho": (9, 13), "Er": (9, 14), "Tm": (9, 15), "Yb": (9, 16), "Lu": (9, 17),
    # Actinides (row 10)
    "Th": (10, 4), "Pa": (10, 5), "U": (10, 6), "Np": (10, 7), "Pu": (10, 8), "Am": (10, 9), "Cm": (10, 10), "Bk": (10, 11), "Cf": (10, 12), "Es": (10, 13), "Fm": (10, 14), "Md": (10, 15), "No": (10, 16), "Lr": (10, 17),
}
# Periodic table is drawn at 1 figure inch = _PT_PX_PER_INCH screen px, so text
# sizes can be set in px and match the st.caption text above it (14px Source
# Sans). Matplotlib's DejaVu Sans has taller capitals (cap height 0.73 em vs
# 0.66 em), so scale it down to the same visual size.
_PT_CELL_PX = 36
_PT_FONT_PX = 14 * 0.66 / 0.73
_PT_PX_PER_INCH = 100
_PT_WIDTH_PX = 18 * _PT_CELL_PX


def _is_hosted_demo() -> bool:
    if os.environ.get("AUTOEMX_DEMO"):
        return True
    # Streamlit Community Cloud mounts the repo here.
    return os.path.isdir("/mount/src")


def _write_upload(upload, dest_dir: Path) -> Path:
    suffix = Path(upload.name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported file '{upload.name}'. Use {', '.join(_ACCEPT)}."
        )
    dest = dest_dir / Path(upload.name).name
    dest.write_bytes(upload.getbuffer())
    return dest


def _failed_result(
    filename: str,
    els_sample,
    els_substrate,
    is_particle: bool,
    error: str,
    error_stage: str,
    reader_report: str | None = None,
) -> SpectrumFitResult:
    return SpectrumFitResult(
        filename=filename,
        energy_keV=np.array([]),
        counts=np.array([]),
        fit=np.array([]),
        background=np.array([]),
        composition_at={},
        composition_wt={},
        els_sample=list(els_sample),
        els_substrate=list(els_substrate),
        is_particle=is_particle,
        error=error,
        error_stage=error_stage,
        reader_report=reader_report,
    )


def _periodic_table_figure(quantifiable: frozenset) -> plt.Figure:
    color_on, color_off = "#1a6bb5", "#d8d8d8"
    text_on, text_off = "white", "#888888"
    pad = 0.08

    cell_in = _PT_CELL_PX / _PT_PX_PER_INCH
    fig = plt.figure(figsize=(18 * cell_in, 10 * cell_in))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(10.5, 0.5)
    ax.axis("off")
    fig.patch.set_alpha(0)

    for symbol, (row, col) in _PT_POSITIONS.items():
        is_q = symbol in quantifiable
        rect = mpatches.FancyBboxPatch(
            (col - 0.5 + pad, row - 0.5 + pad),
            1 - 2 * pad,
            1 - 2 * pad,
            boxstyle="round,pad=0.05",
            linewidth=0,
            facecolor=color_on if is_q else color_off,
        )
        ax.add_patch(rect)
        ax.text(
            col, row, symbol,
            ha="center", va="center",
            fontsize=_PT_FONT_PX * 72 / _PT_PX_PER_INCH,
            color=text_on if is_q else text_off,
            fontweight="bold" if is_q else "normal",
        )

    return fig


def _render_quantifiable_elements_section() -> None:
    """Render an expandable periodic table showing which elements can be quantified."""
    with st.expander("Supported elements for quantification", expanded=True):
        st.caption(
            f"Elements highlighted in blue have peak-to-background standards available "
            f"at {QUANT_BEAM_KV:.0f} kV and can be quantified. "
            f"Grey elements are not supported."
        )
        if QUANTIFIABLE_ELEMENTS:
            fig = _periodic_table_figure(QUANTIFIABLE_ELEMENTS)
            # Rendered at 2x for sharpness, shown at fixed width so it isn't
            # stretched to the (wide-layout) container.
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=2 * _PT_PX_PER_INCH)
            plt.close(fig)
            st.image(buffer.getvalue(), width=_PT_WIDTH_PX)
        else:
            st.info("Standards file could not be read; supported element list unavailable.")


def _results_table(result: SpectrumFitResult):
    rows = []
    elements = list(result.composition_at.keys()) or list(result.composition_wt.keys())
    for el in elements:
        rows.append(
            {
                "Element": el,
                "At%": round(result.composition_at.get(el, 0.0) * 100.0, 2),
                "Wt%": round(result.composition_wt.get(el, 0.0) * 100.0, 2),
            }
        )
    return rows


def main() -> None:
    st.set_page_config(
        page_title="AutoEMX SEM-EDS spectrum quantification",
        layout="wide",
    )
    hosted = _is_hosted_demo()

    st.title("AutoEMX SEM-EDS spectrum quantification")
    st.caption(
        "Scanning Electron Microscopy – Energy-Dispersive X-ray Spectroscopy "
        "(SEM-EDS). Upload one or more SEM-EDS spectra (``.msa``, ``.emsa``, ``.msg``), "
        "quantify them, then save the fitted spectrum as PNG and the composition as TXT."
    )
    st.warning(
        f"**Compositions are valid only for spectra collected at {QUANT_BEAM_KV:.0f} kV.** "
        "The shipped peak-to-background standards are for 15 kV on a ThermoFisher Phenom XL. "
        "A spectrum acquired at any other accelerating voltage can still be fitted, "
        "but the reported composition will be less accurate. The larger the voltage difference, the larger the error. "
        "Spectra from other instruments are still quantifiable, at the expense of accuracy."
    )

    if hosted:
        st.info(
            "This is a **public demo**. Each fit typically takes 0.5–3 minutes, "
            "and the app may sleep after idle time. For production work, install "
            "AutoEMX on your machine (`pip install autoemx`) and run "
            "`python -m autoemx.web`. "
            "**Free use for non-commercial use only.** "
            f"Contact [{_LICENSE_CONTACT}](mailto:{_LICENSE_CONTACT}) "
            "for commercial purposes."
        )
    else:
        st.markdown(
            "Local AutoEMX GUI. You must specify which elements are present — "
            "the engine does not auto-identify unknown peaks."
        )

    st.markdown(
        "**Citation.** If you use this demo or AutoEMX in your research, please cite: "
        "A. Giunto *et al.*, *Accurate SEM‑EDS Quantification, Automation, and "
        "Machine Learning Enable High‑Throughput Compositional Characterization of Powders*, "
        "*Nature Communications* **17**, 9735 (2026). "
        "DOI: [https://doi.org/10.1038/s41467-026-76633-x]"
        "(https://doi.org/10.1038/s41467-026-76633-x)."
    )

    with st.sidebar:
        st.header("Sample")
        els_sample_text = st.text_input(
            "Sample elements",
            value="Bi, Fe, O",
            help="Elements to be quantified in the sample. Comma-separated symbols. Required.",
        )
        els_substrate_text = st.text_input(
            "Substrate elements",
            value="C, O, Al",
            help=(
                "Elements to be fitted, but to be ignored from quantification. "
                "It is recommended not to quantify substrate elements for improved accuracy. "
                "Typical carbon-tape substrate is C, O, Al."
            ),
        )
        is_particle = st.checkbox(
            "Particle / rough sample geometry",
            value=True,
            help="Uncheck if your sample is flat, with roughness lower than 50nm.",
        )
        st.header("Files")
        st.caption(
            f"Collect spectra at **{QUANT_BEAM_KV:.0f} kV**. "
            "Compositions from other voltages are inaccurate."
        )
        uploads = st.file_uploader(
            "SEM-EDS spectra",
            type=[ext.lstrip(".") for ext in _ACCEPT],
            accept_multiple_files=True,
        )
        run = st.button("Quantify", type="primary", use_container_width=True)

    if "results" not in st.session_state:
        st.session_state.results = None

    if not run and st.session_state.results is None:
        st.markdown(
            "Provide elements, upload spectra, then click **Quantify**. "
            f"Spectra **must be collected at {QUANT_BEAM_KV:.0f} kV** for the "
            "composition to be accurate. An example file is "
            "`autoemx/scripts/input/Example_spectrum.msa` (Bi–Fe–O on carbon tape, 15 kV)."
        )
        _render_quantifiable_elements_section()
        return

    if not run:
        results = st.session_state.results
        _render_results(results)
        return

    try:
        els_sample = parse_elements(els_sample_text)
        els_substrate = parse_elements(els_substrate_text)
    except ValueError as exc:
        st.error(str(exc))
        return

    if not els_sample:
        st.error("Enter at least one sample element.")
        return

    try:
        validate_elements_quantifiable(els_sample)
    except ValueError as exc:
        st.error(str(exc))
        _render_quantifiable_elements_section()
        return

    overlap = sorted(set(els_sample) & set(els_substrate))
    if overlap:
        st.warning(
            f"**{', '.join(overlap)}** "
            + ("is" if len(overlap) == 1 else "are")
            + " listed in both sample and substrate elements. "
            "Quantifying an element that is also present in the substrate (e.g. C on carbon tape) "
            "may lead to **major quantification errors** when the substrate contributes substantial "
            "signal for that element. It is recommended to quantify only elements that are not "
            "present in major quantities in the substrate — for example, O from carbon tape is "
            "unlikely to significantly affect the measurement and can generally be quantified with "
            "confidence, while C cannot."
        )

    if not uploads:
        st.error("Upload at least one spectrum file.")
        return

    if hosted and len(uploads) > _DEMO_FILE_CAP:
        st.error(
            f"This demo accepts at most {_DEMO_FILE_CAP} files. "
            "Install AutoEMX locally to quantify larger batches."
        )
        return

    progress = st.progress(0, text="Starting…")
    results: list[SpectrumFitResult] = []
    n_files = len(uploads)

    with tempfile.TemporaryDirectory(prefix="autoemx_web_") as tmp:
        tmp_dir = Path(tmp)
        for i, upload in enumerate(uploads):
            progress.progress(
                i / n_files,
                text=f"Fitting {upload.name} ({i + 1}/{n_files})…",
            )
            try:
                path = _write_upload(upload, tmp_dir)
                result = fit_uploaded_spectrum(
                    path,
                    els_sample=els_sample,
                    els_substrate=els_substrate,
                    is_particle=is_particle,
                )
            except SpectrumReadError as exc:
                result = _failed_result(
                    upload.name,
                    els_sample,
                    els_substrate,
                    is_particle,
                    error=str(exc),
                    error_stage="read",
                    reader_report=exc.report,
                )
            except Exception as exc:
                result = _failed_result(
                    upload.name,
                    els_sample,
                    els_substrate,
                    is_particle,
                    error=str(exc),
                    error_stage="fit",
                )
            results.append(result)
        progress.progress(1.0, text="Done.")

    st.session_state.results = results
    _render_results(results)


def _render_results(results: list[SpectrumFitResult]) -> None:
    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    st.success(f"Finished {len(ok)} of {len(results)} spectrum(s).")
    if failed:
        st.warning(
            "Failed: " + ", ".join(f"{r.filename} ({r.error})" for r in failed)
        )

    if ok:
        st.download_button(
            "Download all compositions (TXT)",
            data=format_batch_composition_txt(ok),
            file_name="autoemx_compositions.txt",
            mime="text/plain",
        )

    for result in results:
        st.divider()
        st.subheader(result.filename)
        if result.error:
            st.error(result.error)
            if result.error_stage == "read":
                _render_reader_report(result)
            continue

        metrics = st.columns(3)
        metrics[0].metric(
            "R²",
            f"{result.r_squared:.5f}" if result.r_squared is not None else "—",
        )
        metrics[1].metric(
            "Reduced χ²",
            f"{result.reduced_chi_sq:.1f}" if result.reduced_chi_sq is not None else "—",
        )
        metrics[2].metric(
            "Analytical error",
            (
                f"{result.analytical_error * 100:.2f} w%"
                if result.analytical_error is not None
                else "—"
            ),
        )
        if result.beam_energy_kV is not None:
            if beam_energy_supports_quantification(result.beam_energy_kV):
                st.caption(
                    f"Header beam energy: {result.beam_energy_kV:.3g} kV "
                    f"(matches the required {QUANT_BEAM_KV:.0f} kV)."
                )
            else:
                st.error(
                    f"This spectrum was collected at **{result.beam_energy_kV:.3g} kV**, "
                    f"not {QUANT_BEAM_KV:.0f} kV. The composition below is **not valid** "
                    "with the shipped 15 kV standards."
                )

        fig = fitted_spectrum_figure(result)
        st.pyplot(fig, clear_figure=False)
        png = figure_to_png_bytes(fig)
        fig.clf()

        table = _results_table(result)
        if table:
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.info("No composition values were returned for this spectrum.")

        col_png, col_txt = st.columns(2)
        stem = Path(result.filename).stem
        with col_png:
            st.download_button(
                "Save fitted spectrum (PNG)",
                data=png,
                file_name=f"{stem}_fitted.png",
                mime="image/png",
                key=f"png-{result.filename}",
            )
        with col_txt:
            st.download_button(
                "Save composition (TXT)",
                data=format_composition_txt(result),
                file_name=f"{stem}_composition.txt",
                mime="text/plain",
                key=f"txt-{result.filename}",
            )


def _render_reader_report(result: SpectrumFitResult) -> None:
    st.markdown(
        "The EMSA/MSA reader could not parse this file. Download a report and "
        f"email it to **{REPORT_EMAIL}** so the reader can be updated for this variant. "
        "Attach the original spectrum as well if you can share it."
    )
    report = result.reader_report or result.error or ""
    stem = Path(result.filename).stem
    col_dl, col_mail = st.columns(2)
    with col_dl:
        st.download_button(
            "Download reader report",
            data=report,
            file_name=f"{stem}_emsa_reader_report.txt",
            mime="text/plain",
            key=f"reader-report-{result.filename}",
        )
    with col_mail:
        st.link_button(
            "Open email to maintainer",
            mailto_reader_report_url(result.filename, result.error or ""),
        )
    st.caption(
        f"You can also open a GitHub issue at {GITHUB_NEW_ISSUE_URL} "
        "and attach the same report."
    )


main()
