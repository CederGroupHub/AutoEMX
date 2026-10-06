#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Periodic table of the elements quantifiable with the P/B standards of a microscope (AutoEMX GUI)."""

from __future__ import annotations

from typing import Iterable, List, Optional

from dash import html

# (row, col) positions in the standard 18-column periodic table layout.
# Lanthanides and actinides are placed in rows 9 and 10 (with a gap after row 7).
PT_POSITIONS: dict[str, tuple[int, int]] = {
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


def periodic_table(quantifiable: Optional[Iterable[str]], sample_elements: Iterable[str] = ()) -> html.Div:
    """
    Periodic table: quantifiable elements in blue, others in grey. Elements of the samples are outlined,
    in red when they cannot be quantified. ``quantifiable`` None: no standards.
    """
    quantifiable = set(quantifiable or [])
    sample_elements = set(sample_elements)
    cells: List[html.Div] = []
    for symbol, (row, col) in PT_POSITIONS.items():
        cls = "pt-cell" + (" pt-quant" if symbol in quantifiable else "")
        if symbol in sample_elements:
            cls += " pt-sample" if symbol in quantifiable else " pt-missing"
        title = symbol + (" · quantifiable" if symbol in quantifiable else " · no standard")
        cells.append(html.Div(symbol, className=cls, title=title,
                              style={"gridRow": str(row), "gridColumn": str(col)}))
    return html.Div(cells, className="pt-grid")
