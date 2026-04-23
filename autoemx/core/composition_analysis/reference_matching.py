#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference matching module for phase identification and mixture analysis."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import autoemx.utils.constants as cnst


class ReferenceMatchingModule:
	"""Container for reference matching algorithms extracted from the analyser."""

	def _correlate_centroids_to_refs(
		self,
		centroids: 'np.ndarray',
		cluster_radii: 'np.ndarray',
		ref_phases_df: 'pd.DataFrame'
	) -> Tuple[List[float], 'pd.DataFrame']:
		all_ref_phases = ref_phases_df.to_numpy()
		refs_dict = []
		max_raw_confs = []

		for centroid, radius in zip(centroids, cluster_radii):
			distances = np.linalg.norm(all_ref_phases - centroid, axis=1)
			indices = np.where(distances < max(0.1, 5 * radius))[0]
			ref_names = [self.ref_formulae[i] for i in indices]
			ref_phases = [all_ref_phases[i] for i in indices]
			max_raw_conf, refs_dict_row = ReferenceMatchingModule._get_ref_confidences(
				centroid, ref_phases, ref_names
			)
			max_raw_confs.append(max_raw_conf)
			refs_dict.append(refs_dict_row)

		refs_assigned_df = pd.DataFrame(refs_dict)
		return max_raw_confs, refs_assigned_df

	def _assign_reference_phases(self, centroids, rms_dist_cluster):
		min_conf = None
		max_raw_confs = None
		refs_assigned_df = None
		if self.ref_formulae is not None:
			max_raw_confs, refs_assigned_df = self._correlate_centroids_to_refs(
				centroids, rms_dist_cluster, self.ref_phases_df
			)
			if len(max_raw_confs) > 0:
				max_confs_num = [conf for conf in max_raw_confs if conf is not None]
				if len(max_confs_num) > 0:
					min_conf = min(max_confs_num)
		return min_conf, max_raw_confs, refs_assigned_df

	@staticmethod
	def _get_ref_confidences(
		centroid: 'np.ndarray',
		ref_phases: 'np.ndarray',
		ref_names: List[str]
	) -> Tuple[Optional[float], Dict]:
		"""Compute raw and weighted confidence scores for candidate references."""
		if ref_phases == [] or len(ref_phases) == 0:
			refs_dict = {
				f'{cnst.CND_DF_KEY}1': np.nan,
				f'{cnst.CS_RAW_CND_DF_KEY}1': np.nan,
				f'{cnst.CS_CND_DF_KEY}1': np.nan,
			}
			max_raw_conf = None
		else:
			distances = np.linalg.norm(ref_phases - centroid, axis=1)
			raw_confidences = np.exp(-distances**2 / (2 * 0.03**2))

			weights_conf = np.exp(-(1 - raw_confidences)**2 / (2 * 0.3**2))
			weights_conf /= np.sum(weights_conf)
			confidences = raw_confidences * weights_conf

			max_raw_conf = float(np.max(raw_confidences))

			sorted_indices = np.argsort(-confidences)
			sorted_ref_names = np.array(ref_names)[sorted_indices]
			sorted_confidences = confidences[sorted_indices]
			sorted_raw_confs = raw_confidences[sorted_indices]

			refs_dict = {}
			for i, (ref_name, conf, conf_raw) in enumerate(
				zip(sorted_ref_names, sorted_confidences, sorted_raw_confs)
			):
				if conf_raw > 0.05:
					refs_dict[f'{cnst.CND_DF_KEY}{i+1}'] = ref_name
					refs_dict[f'{cnst.CS_CND_DF_KEY}{i+1}'] = np.round(conf, 2)
					refs_dict[f'{cnst.CS_RAW_CND_DF_KEY}{i+1}'] = np.round(conf_raw, 2)

		return max_raw_conf, refs_dict
