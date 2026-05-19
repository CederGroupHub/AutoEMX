#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update EDS standards JSON file in place.

This script allows you to:
- Delete entries by ID and el_line (or "all")
- Exclude entries from mean calculation by ID and el_line (or "all")
- Recompute the means using only entries with Use_for_mean_calc = True

Usage:
    - Set the variables below as needed.
    - Run the script.

Created on Tue May 19 2026
@author: Andrea
"""

from pathlib import Path
import json
import numpy as np

# =============================================================================
# Input
# =============================================================================

# Path to the EDS standards JSON file
standards_json_path = '/absolute/path/to/EDS_Stds_15keV.json'  # <-- Set this
standards_json_path = '/Users/Andrea_1/Desktop/Work/Projects/SEM EDX automation/EDX standards/Auto measurements/EDS_Stds_15keV.json'


# List of dicts: {ID: el_line(s) or "all"} to delete
# Example: [{"BaSO4": "all"}, {"BaSO4": ["Ba_La1", "O_Ka1"]}]
delete_std_lines = None  # or []

# List of dicts: {ID: el_line(s) or "all"} to exclude from mean calc
# Example: [{"BaSO4": "all"}, {"BaSO4": ["Ba_La1", "O_Ka1"]}]
exclude_std_lines_from_mean_calc = None  # or []

# =============================================================================
# Run
# =============================================================================

def build_lookup(list_of_dicts):
    from collections import defaultdict
    lookup = defaultdict(set)
    if not list_of_dicts:
        return lookup
    for d in list_of_dicts:
        for k, v in d.items():
            if v == "all":
                lookup[k] = "all"
            elif isinstance(v, list):
                lookup[k].update(v)
            else:
                lookup[k].add(v)
    return lookup


def update_standards_file(
    file_path,
    delete_std_lines=None,
    exclude_std_lines_from_mean_calc=None
):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    delete_lookup = build_lookup(delete_std_lines)
    exclude_lookup = build_lookup(exclude_std_lines_from_mean_calc)


    standards = data["standards_by_mode"]["point"]
    el_lines_to_remove = []
    for el_line, std in standards.items():
        new_entries = []
        for entry in std["entries"]:
            entry_id = entry.get("ID")
            # Deletion
            if entry_id in delete_lookup:
                if delete_lookup[entry_id] == "all" or el_line in delete_lookup[entry_id]:
                    continue  # skip (delete)
            # Exclusion from mean calc
            if entry_id in exclude_lookup:
                if exclude_lookup[entry_id] == "all" or el_line in exclude_lookup[entry_id]:
                    entry["Use_for_mean_calc"] = False
            new_entries.append(entry)
        std["entries"] = new_entries

        # If no entries remain, mark for removal
        if len(std["entries"]) == 0:
            el_lines_to_remove.append(el_line)
            continue

        # Recompute means
        mean_fields = [
            "Corrected_PB", "Stdev_PB", "Rel_stdev_PB (%)", "Measured_PB"
        ]
        mean_entry = {k: [] for k in mean_fields}
        datetimes = []
        for entry in std["entries"]:
            if entry.get("Use_for_mean_calc", True):
                for k in mean_fields:
                    if k in entry:
                        mean_entry[k].append(entry[k])
                if "datetime" in entry:
                    datetimes.append(entry["datetime"])
        reference_mean = {}
        if mean_entry["Corrected_PB"]:
            reference_mean["Corrected_PB"] = float(np.mean(mean_entry["Corrected_PB"]))
            reference_mean["Stdev_PB"] = float(np.std(mean_entry["Corrected_PB"]))
            if reference_mean["Corrected_PB"] != 0:
                reference_mean["Rel_stdev_PB (%)"] = (
                    reference_mean["Stdev_PB"] / reference_mean["Corrected_PB"] * 100
                )
            else:
                reference_mean["Rel_stdev_PB (%)"] = 0.0
            reference_mean["datetime"] = datetimes[-1] if datetimes else None
            std["reference_mean"] = reference_mean
        else:
            # No valid entries for mean: remove reference_mean if present
            if "reference_mean" in std:
                del std["reference_mean"]

    # Remove el_lines with no entries
    for el_line in el_lines_to_remove:
        del standards[el_line]

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    update_standards_file(
        standards_json_path,
        delete_std_lines=delete_std_lines,
        exclude_std_lines_from_mean_calc=exclude_std_lines_from_mean_calc
    )
    print(f"Updated standards file: {standards_json_path}")
