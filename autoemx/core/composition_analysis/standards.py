#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental standards mixin for PB ratio workflows."""

import json
import os
import shutil
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

import autoemx.calibrations as calibs
import autoemx.utils.constants as cnst
from autoemx.core.quantifier import Quant_Corrections, XSp_Quantifier
from autoemx.utils import make_unique_path, print_double_separator, print_single_separator

from autoemx._logging import get_logger
logger = get_logger(__name__)


class StandardsModule:
    def _compile_standards_from_references(self) -> dict:
        std_dict_all_modes, stds_filepath = self._load_xsp_standards()
        std_dict_all_lines = std_dict_all_modes[self.measurement_cfg.mode]
        ref_lines = XSp_Quantifier.xray_quant_ref_lines
        ref_formulae = self.clustering_cfg.ref_formulae

        filtered_std_dict = {}
        for el in self.detectable_els_sample:
            for line in ref_lines:
                el_line = f"{el}_{line}"
                if el_line not in std_dict_all_lines:
                    continue

                ref_entries = []
                for i, std_dict in enumerate(std_dict_all_lines[el_line]):
                    if std_dict[cnst.STD_ID_KEY] != cnst.STD_MEAN_ID_KEY:
                        try:
                            std_comp = Composition(std_dict[cnst.STD_FORMULA_KEY])
                            for ref_formula in ref_formulae:
                                if std_comp.reduced_formula == Composition(ref_formula).reduced_formula:
                                    ref_entries += [i]
                        except Exception:
                            pass
                    else:
                        std_mean_value = std_dict[cnst.COR_PB_DF_KEY]

                if len(ref_entries) < 1 and not self.exp_stds_cfg.is_exp_std_measurement:
                    ref_value = std_mean_value
                else:
                    new_std_ref_list = [std_d for i, std_d in enumerate(std_dict_all_lines[el_line]) if i in ref_entries]
                    list_pb = [ref_line[cnst.COR_PB_DF_KEY] for ref_line in new_std_ref_list]
                    ref_value = float(np.mean(list_pb))

                filtered_std_dict[el_line] = [{
                    cnst.STD_ID_KEY: cnst.STD_MEAN_ID_KEY,
                    cnst.COR_PB_DF_KEY: ref_value,
                }]

        return filtered_std_dict

    def _fit_stds_and_save_results(self, backup_previous_data: bool = True) -> Union[Tuple, None]:
        std_ref_lines = None
        results_df = None
        z_sample = None

        self._fit_and_quantify_spectra(quantify=False)
        data_df = self._save_collected_data(None, None, backup_previous_data=backup_previous_data, include_spectral_data=True)

        if data_df is not None and not data_df.empty:
            std_ref_lines = self._assemble_std_PB_data(data_df)
            if std_ref_lines != {}:
                pb_corrected, z_sample = self._calc_corrected_PB(std_ref_lines)
                results_df = self._save_std_results(std_ref_lines, pb_corrected)
            return std_ref_lines, results_df, z_sample

        return None, None, None

    def _evaluate_exp_std_fit(self, tot_n_spectra: int) -> Tuple[bool, bool]:
        is_fit_successful = False
        is_converged = False

        if self.verbose:
            print_double_separator()
            logger.info(f"🔬 Fitting after collection of {tot_n_spectra} spectra...")

        _, results_df, _ = self._fit_stds_and_save_results(backup_previous_data=False)
        if results_df is not None and not results_df.empty:
            is_fit_successful = True
            num_valid_spectra = int(np.min(results_df[cnst.N_SP_USED_KEY]))
            is_converged = num_valid_spectra >= self.min_n_spectra

        return is_fit_successful, is_converged

    def _assemble_std_PB_data(
        self,
        data_df: "pd.DataFrame"
    ) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        data_df = data_df.dropna(axis=1, how="all")
        data_filtered_df = data_df[data_df[cnst.QUANT_FLAG_DF_KEY].isin(self.exp_stds_cfg.quant_flags_accepted)]

        all_fitted_el_lines = [el_line for el_line in self._th_peak_energies.keys() if el_line in data_filtered_df.columns]
        fitted_std_el_lines = [el_line for el_line in all_fitted_el_lines if el_line.split("_")[0] in self.detectable_els_sample]

        std_ref_lines = {}
        for el_line in fitted_std_el_lines:
            meas_pb_ratios = data_filtered_df[el_line].tolist()
            if len(meas_pb_ratios) > 0:
                std_ref_lines[el_line] = {
                    cnst.PB_RATIO_KEY: meas_pb_ratios,
                    cnst.MEAN_PB_KEY: float(np.nanmean(meas_pb_ratios)),
                    cnst.STDEV_PB_DF_KEY: float(np.nanstd(meas_pb_ratios)),
                    cnst.PEAK_TH_ENERGY_KEY: self._th_peak_energies[el_line],
                }

        return std_ref_lines

    def _calc_corrected_PB(
        self,
        std_ref_lines: Dict[str, Dict[str, Union[float, List[float]]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        peak_energies_dict: Dict[str, float] = {}
        means_pb: List[float] = []
        w_frs: List[float] = []

        for el_line, el_line_dict in std_ref_lines.items():
            peak_energies_dict[el_line] = el_line_dict[cnst.PEAK_TH_ENERGY_KEY]
            means_pb.append(float(el_line_dict[cnst.MEAN_PB_KEY]))
            el = el_line.split('_')[0]
            w_frs.append(self.exp_stds_cfg.w_frs[el])

        zaf_calculator = Quant_Corrections(
            elements=self.detectable_els_sample,
            beam_energy=self.measurement_cfg.beam_energy_keV,
            emergence_angle=self.measurement_cfg.emergence_angle,
            meas_mode=self.measurement_cfg.mode,
            verbose=False,
        )

        nominal_w_frs = [self.exp_stds_cfg.w_frs[el] for el in self.detectable_els_sample]
        zaf_pb, z_sample = zaf_calculator.get_ZAF_mult_f_pb(nominal_w_frs, peak_energies_dict)
        pb_corrected = zaf_pb * np.array(means_pb) / np.array(w_frs)
        return pb_corrected, z_sample

    def _save_std_results(
        self,
        std_ref_lines: Dict[str, Dict[str, Any]],
        pb_corrected: List[float]
    ) -> Optional[pd.DataFrame]:
        if not std_ref_lines:
            return None

        means_pb = []
        stdevs_pb = []
        n_spectra_per_line = []
        line_keys = list(std_ref_lines.keys())

        for el_line in line_keys:
            el_line_dict = std_ref_lines[el_line]
            means_pb.append(el_line_dict[cnst.MEAN_PB_KEY])
            stdevs_pb.append(el_line_dict[cnst.STDEV_PB_DF_KEY])
            pb_ratios = el_line_dict[cnst.PB_RATIO_KEY]
            n_spectra_used = sum((x is not None) and (not (isinstance(x, float) and np.isnan(x))) for x in pb_ratios)
            n_spectra_per_line.append(n_spectra_used)

        results_df = pd.DataFrame({
            cnst.MEAS_PB_DF_KEY: means_pb,
            cnst.STDEV_PB_DF_KEY: stdevs_pb,
            cnst.COR_PB_DF_KEY: pb_corrected,
            cnst.REL_ER_PERCENT_PB_DF_KEY: np.array(stdevs_pb) / np.array(means_pb) * 100,
            cnst.N_SP_USED_KEY: n_spectra_per_line,
        }, index=line_keys)

        filename = f"{cnst.STDS_RESULT_FILENAME}_{self.measurement_cfg.mode}" + self.output_filename_suffix
        results_path = os.path.join(self.sample_result_dir, filename + '.csv')
        results_df.to_csv(results_path, index=True, header=True)
        return results_df

    def _load_xsp_standards(self) -> Tuple[dict, str]:
        meas_mode = self.measurement_cfg.mode
        update_separate_std_dict = self.exp_stds_cfg.is_exp_std_measurement and self.exp_stds_cfg.generate_separate_std_dict

        if self.standards_dict is None:
            std_f_dir = None
            if update_separate_std_dict or self.quant_cfg.use_project_specific_std_dict:
                project_dir = os.path.dirname(self.sample_result_dir)
                std_f_dir = project_dir

            try:
                standards, stds_filepath = calibs.load_standards(self.measurement_cfg.type, self.measurement_cfg.beam_energy_keV, std_f_dir=std_f_dir)
            except FileNotFoundError:
                stds_filepath = calibs.standards_dir
                standards = {meas_mode: {}}
            else:
                if update_separate_std_dict and os.path.dirname(stds_filepath) != project_dir:
                    stds_filepath = shutil.copy(stds_filepath, project_dir)
        else:
            standards = self.standards_dict
            stds_filepath = ''

        if meas_mode not in standards:
            standards[meas_mode] = {}

        return standards, stds_filepath

    def _update_standard_library(
        self,
        std_ref_lines: Dict[str, Dict[str, Union[float, List[float]]]],
        results_df: pd.DataFrame,
        z_sample: np.ndarray
    ) -> None:
        meas_mode = self.measurement_cfg.mode
        if self.standards_dict is not None:
            self.standards_dict = None
        standards, stds_filepath = self._load_xsp_standards()
        std_lib = standards[meas_mode]

        for el_line, stds_list in list(std_lib.items()):
            for i, std_dict in enumerate(list(stds_list)):
                if std_dict.get(cnst.STD_ID_KEY) == self.sample_cfg.ID:
                    std_lib[el_line].pop(i)
                    break

        now = datetime.now()
        for el_line in std_ref_lines.keys():
            std_dict_new = {
                cnst.STD_ID_KEY: self.sample_cfg.ID,
                cnst.STD_FORMULA_KEY: self.exp_stds_cfg.formula,
                cnst.STD_TYPE_KEY: self.sample_cfg.type,
                cnst.DATETIME_KEY: now.strftime("%Y-%m-%d %H:%M:%S"),
                cnst.COR_PB_DF_KEY: results_df.at[el_line, cnst.COR_PB_DF_KEY],
                cnst.MEAS_PB_DF_KEY: results_df.at[el_line, cnst.MEAS_PB_DF_KEY],
                cnst.STDEV_PB_DF_KEY: results_df.at[el_line, cnst.STDEV_PB_DF_KEY],
                cnst.REL_ER_PERCENT_PB_DF_KEY: results_df.at[el_line, cnst.REL_ER_PERCENT_PB_DF_KEY],
                cnst.STD_USE_FOR_MEAN_KEY: self.exp_stds_cfg.use_for_mean_PB_calc,
                cnst.STD_Z_KEY: z_sample,
            }

            if el_line in std_lib:
                std_lib[el_line].append(std_dict_new)
            else:
                std_lib[el_line] = [std_dict_new]

            std_el_line_entries = [std for std in std_lib[el_line] if std.get(cnst.STD_ID_KEY) != cnst.STD_MEAN_ID_KEY]
            list_pb_for_mean = [std[cnst.COR_PB_DF_KEY] for std in std_el_line_entries if std[cnst.STD_USE_FOR_MEAN_KEY]]
            if len(list_pb_for_mean) > 0:
                mean_pb = float(np.mean(list_pb_for_mean))
                stddev_mean_pb = float(np.std(list_pb_for_mean))
                error_mean_pb = (stddev_mean_pb / mean_pb * 100) if mean_pb else float("nan")
                std_dict_mean = {
                    cnst.STD_ID_KEY: cnst.STD_MEAN_ID_KEY,
                    cnst.DATETIME_KEY: now.strftime("%Y-%m-%d %H:%M:%S"),
                    cnst.COR_PB_DF_KEY: mean_pb,
                    cnst.STDEV_PB_DF_KEY: stddev_mean_pb,
                    cnst.REL_ER_PERCENT_PB_DF_KEY: error_mean_pb,
                }
                std_el_line_entries.append(std_dict_mean)
            std_lib[el_line] = std_el_line_entries

        with open(stds_filepath, "w") as file:
            json.dump(standards, file, indent=2)
