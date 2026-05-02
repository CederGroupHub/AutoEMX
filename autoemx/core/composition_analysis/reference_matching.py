#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference matching module for phase identification and mixture analysis."""

import itertools
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

import autoemx.utils.constants as cnst
from autoemx.core.composition_analysis.plotting import PlottingModule


class ReferenceMatchingModule:
	"""Container for reference matching algorithms extracted from the analyser."""

	def _correlate_centroids_to_refs(
		self: Any,
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
			max_raw_confs, refs_assigned_df = ReferenceMatchingModule._correlate_centroids_to_refs(
				self,
				centroids,
				rms_dist_cluster,
				self.ref_phases_df,
			)
			if len(max_raw_confs) > 0:
				max_confs_num = [c for c in max_raw_confs if isinstance(c, (int, float))]
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

	# =========================================================================
	# Mixture analysis / NMF
	# =========================================================================

	def _assign_mixtures(self, k, labels, compositions_df, rms_dist_cluster, max_raw_confs, n_points_per_cluster):
		"""
		Determine if clusters are mixtures or single phases, using candidate phases and NMF if needed.

		Returns
		-------
		clusters_assigned_mixtures : list
			List of mixture assignments for each cluster.
			
		Potential improvements
		----------------------
		Instead of using the cluster standard deviation, use covariance of elemental fractions
		to discern clusters that may originate from binary phase mixtures or solid solutions.
		"""
		clusters_assigned_mixtures = []
		for i in range(k):
			# Get compositions of data points included in cluster as np.array (only detectable elements)
			cluster_data = compositions_df[self.detectable_els_sample].iloc[labels == i].values
			max_mix_conf = 0
			mixtures_dicts = []

			max_rmsdist_single_cluster = 0.03
			if rms_dist_cluster[i] < max_rmsdist_single_cluster:
				if max_raw_confs is None or len(max_raw_confs) < 1:
					is_cluster_single_phase = n_points_per_cluster[i] > 3
				elif max_raw_confs[i] is not None and max_raw_confs[i] > 0.5:
					is_cluster_single_phase = True
				else:
					is_cluster_single_phase = False
			else:
				is_cluster_single_phase = False

			if is_cluster_single_phase:
				# Cluster determined to stem from a single phase
				pass
			elif len(self.ref_formulae) > 1:
				max_mix_raw_conf, mixtures_dicts = ReferenceMatchingModule._identify_mixture_from_refs(self, cluster_data, cluster_ID=i)
				max_mix_conf = max(max_mix_conf, max_mix_raw_conf)
			if not is_cluster_single_phase and max_mix_conf < 0.5:
				mix_nmf_conf, mixture_dict = ReferenceMatchingModule._identify_mixture_nmf(self, cluster_data, cluster_ID=i)
				if mixture_dict is not None:
					mixtures_dicts.append(mixture_dict)
				max_mix_conf = max(max_mix_conf, mix_nmf_conf)
			clusters_assigned_mixtures.append(mixtures_dicts)
		return clusters_assigned_mixtures


	def _identify_mixture_from_refs(self, X: 'np.ndarray', cluster_ID: int = None) -> Tuple[float, List[Dict]]:
		"""
		Identify mixtures within a cluster by testing all pairs of candidate phases using constrained optimization.

		For each possible pair of candidate phases, tests if the cluster compositions (X)
		can be well described by a linear combination of the two candidate phases, using
		non-negative matrix factorization (NMF) with fixed bases.

		Parameters
		----------
		X : np.ndarray
			Cluster data (compositions), shape (n_samples, n_features).
		cluster_ID : int
			Current cluster ID. Used for violin plot name

		Returns
		-------
		max_confidence : float
			The highest confidence score among all tested mixtures.
		mixtures_dicts : list of Dict
			List of mixture descriptions for all successful reference pairs.
		cluster_ID : int
			Current cluster ID. Used for violin plot name
		"""
		# Generate all possible pairs of candidate phases
		ref_pair_combinations = list(itertools.combinations(range(len(self.ref_phases_df)), 2))

		mixtures_dicts = []
		max_confidence = 0

		for ref_comb in ref_pair_combinations:
			# Get the names of the candidate phases in this pair
			ref_names = [self.ref_formulae[ref_i] for ref_i in ref_comb]

			# Ratio of weights of references, for molar concentrations of parent phases
			ref_w_r = self.ref_weights_in_mixture[ref_comb[0]] / self.ref_weights_in_mixture[ref_comb[1]]

			# Get matrix of basis vectors (H) for the two candidate phases
			H = np.array([
				self.ref_phases_df[self.detectable_els_sample].iloc[ref_i].values
				for ref_i in ref_comb
			])

			# Perform NMF with fixed H to fit the cluster data as a mixture of the two candidate phases
			W, _ = ReferenceMatchingModule._nmf_with_constraints(self, X, n_components=2, fixed_H=H)

			# Compute reconstruction error for the fit
			recon_er = ReferenceMatchingModule._calc_reconstruction_error(self, X, W, H)

			# If the pair yields an acceptable reconstruction error, store the result
			pair_dict, conf = ReferenceMatchingModule._get_mixture_dict_with_conf(self, W, ref_w_r, recon_er, ref_names, cluster_ID)
			if pair_dict is not None:
				mixtures_dicts.append(pair_dict)
				max_confidence = max(max_confidence, conf)

		return max_confidence, mixtures_dicts


	def _calc_reconstruction_error(
		self,
		X: 'np.ndarray',
		W: 'np.ndarray',
		H: 'np.ndarray'
	) -> float:
		"""
		Calculate the reconstruction error for a matrix factorization X ≈ W @ H.

		The error metric is an exponential penalty (with parameter alpha) applied to the
		absolute difference between X and its reconstruction W @ H, normalized by the
		number of elements in X. This penalizes large deviations more strongly.
		"""
		WH = np.dot(W, H)
		alpha = 15
		norm = np.sum(np.exp(alpha * np.abs(X - WH)) - 1)
		m, n = X.shape
		normalized_norm = norm / (m * n)
		return normalized_norm


	def _get_mixture_dict_with_conf(
		self,
		W: 'np.ndarray',
		ref_w_r: float,
		reconstruction_error: float,
		ref_names: List[str],
		cluster_ID: int = None
	) -> Tuple[Optional[Dict], float]:
		"""
		Evaluate if a cluster is a mixture of two candidate phases, and compute a confidence score.

		If the reconstruction error is below a set threshold, computes a confidence score and
		transforms the NMF coefficients into molar fractions. Returns a dictionary describing
		the mixture and the confidence score.
		"""
		min_acceptable_recon_error = 2  # Empirically determined

		save_violin_plot = getattr(
			self.powder_meas_cfg,
			"is_known_powder_mixture_meas",
			False,
		)

		if reconstruction_error < min_acceptable_recon_error or save_violin_plot:
			gauss_sigma = 0.5
			conf = np.exp(-reconstruction_error**2 / (2 * gauss_sigma**2))

			W_mol_frs = []
			for c1, c2 in W:
				x2 = c2 * ref_w_r / (1 - c2 * (1 - ref_w_r))
				x1 = c1 * (1 + x2 * (1 / ref_w_r - 1))
				W_mol_frs.append([x1, x2])
			W_mol_frs = np.array(W_mol_frs)

			mol_frs_norm_means = np.mean(W_mol_frs, axis=0)
			mol_frs_norm_stddevs = np.std(W_mol_frs, axis=0)

			if save_violin_plot:
				PlottingModule._save_violin_plot_powder_mixture(self, W_mol_frs, ref_names, cluster_ID)

			mixture_dict = {
				cnst.REF_NAME_KEY: ref_names,
				cnst.CONF_SCORE_KEY: conf,
				cnst.MOLAR_FR_MEAN_KEY: mol_frs_norm_means[0],
				cnst.MOLAR_FR_STDEV_KEY: mol_frs_norm_stddevs[0]
			}
		else:
			mixture_dict = None
			conf = 0

		return mixture_dict, conf


	def _nmf_with_constraints(
		self,
		X: 'np.ndarray',
		n_components: int,
		fixed_H: 'np.ndarray' = None
	) -> Tuple['np.ndarray', 'np.ndarray']:
		"""
		Perform Non-negative Matrix Factorization (NMF) with optional constraints on the factor matrices.

		This function alternates between optimizing two non-negative matrices W and H, such that X ≈ W @ H:
		  - If H is fixed (provided via fixed_H), only W is updated.
		  - If H is not fixed, both W and H are updated via alternating minimization.

		Constraints:
		  - Both W and H are non-negative.
		  - The rows of both W (sum of coefficients) and H (sum of elemental fractions) sum to 1.
		  - Sparsity regularization (L1) is applied to H when it is updated.
		"""
		max_iter = 1000
		convergence_tol = 1e-3
		lambda_H = 0

		W = np.random.rand(X.shape[0], n_components)
		if fixed_H is None:
			H = np.random.rand(n_components, X.shape[1])
		else:
			H = fixed_H

		prev_W, prev_H = np.inf, np.inf
		convergence = np.inf
		i = 0

		while convergence > convergence_tol and i < max_iter:
			W_var = cp.Variable((X.shape[0], n_components), nonneg=True)
			objective_W = cp.Minimize(cp.sum_squares(X - W_var @ H))
			constraints_W = [cp.sum(W_var, axis=1) == 1]
			problem_W = cp.Problem(objective_W, constraints_W)
			problem_W.solve(solver=cp.ECOS)
			W = W_var.value

			if fixed_H is None:
				H_var = cp.Variable((n_components, X.shape[1]), nonneg=True)
				objective_H = cp.Minimize(
					cp.sum_squares(X - W @ H_var) + lambda_H * cp.norm1(H_var)
				)
				constraints_H = [cp.sum(H_var, axis=1) == 1]
				problem_H = cp.Problem(objective_H, constraints_H)
				problem_H.solve(solver=cp.ECOS)
				H = H_var.value

			convergence_W = np.linalg.norm(W - prev_W, 'fro')
			convergence_H = np.linalg.norm(H - prev_H, 'fro') if fixed_H is None else 0
			convergence = max(convergence_W, convergence_H)

			prev_W, prev_H = W, H
			i += 1

		return W, H


	def _identify_mixture_nmf(
		self,
		X: 'np.ndarray',
		n_components: int = 2,
		cluster_ID: int = None
	) -> Tuple[float, Optional[Dict]]:
		"""
		Identify a mixture within a cluster using unconstrained NMF (Non-negative Matrix Factorization).

		This method fits the cluster data X to n_components using NMF with constraints (rows of W and H sum to 1),
		evaluates the reconstruction error, and if acceptable, returns a dictionary describing the mixture and a confidence score.
		"""
		mixture_dict = None
		conf = 0

		W, H = ReferenceMatchingModule._nmf_with_constraints(self, X, n_components)
		recon_er = ReferenceMatchingModule._calc_reconstruction_error(self, X, W, H)
		ref_names, ref_weights = ReferenceMatchingModule._get_pretty_formulas_nmf(self, H, n_components)
		ref_w_r = ref_weights[0] / ref_weights[1]
		mixture_dict, conf = ReferenceMatchingModule._get_mixture_dict_with_conf(self, W, ref_w_r, recon_er, ref_names, cluster_ID)

		return conf, mixture_dict


	def _get_pretty_formulas_nmf(
		self,
		phases: 'np.ndarray',
		n_components: int
	) -> Tuple[List[str], List[float]]:
		"""
		Generate human-readable (pretty) formulas from NMF bases, accounting for data noise.

		For each component, filters out small fractions, constructs a composition dictionary,
		and returns a formula string and a weight or atom count, depending on the clustering feature.
		"""
		ref_names = []
		ref_weights = []

		for i in range(n_components):
			frs = phases[i, :].copy()
			frs[frs < 0.005] = 0

			fr_dict = {el: fr for el, fr in zip(self.detectable_els_sample, frs)}

			if self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
				comp = Composition().from_weight_dict(fr_dict)
			elif self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
				comp = Composition(fr_dict)

			formula = comp.get_integer_formula_and_factor()[0]
			ref_integer_comp = Composition(formula)
			min_at_n = min(ref_integer_comp.get_el_amt_dict().values())
			pretty_at_frs = {el: round(n / min_at_n, 1) for el, n in ref_integer_comp.get_el_amt_dict().items()}
			pretty_comp = Composition(pretty_at_frs)
			pretty_formula = pretty_comp.formula
			ref_names.append(pretty_formula)

			if self.clustering_cfg.features == cnst.W_FR_CL_FEAT:
				ref_weights.append(pretty_comp.weight)
			elif self.clustering_cfg.features == cnst.AT_FR_CL_FEAT:
				n_atoms_in_formula = sum(pretty_comp.get_el_amt_dict().values())
				ref_weights.append(n_atoms_in_formula)

		return ref_names, ref_weights


	def _build_mixtures_df(
		self,
		clusters_assigned_mixtures: List[List[Dict]]
	) -> 'pd.DataFrame':
		"""
		Build a DataFrame summarizing mixture assignments for each cluster.

		For each cluster, sorts mixture dictionaries by confidence score and extracts:
		  - candidate phase names (as a comma-separated string)
		  - Confidence score
		  - Molar ratio (mean / (1 - mean))
		  - Mean and standard deviation of the main component's molar fraction
		"""
		mixtures_strings_dict = []
		for mixtures_dict in clusters_assigned_mixtures:
			if mixtures_dict:
				sorted_mixtures = sorted(mixtures_dict, key=lambda x: -x[cnst.CONF_SCORE_KEY])
				cluster_mix_dict = {}
				for i, mixture_dict in enumerate(sorted_mixtures, start=1):
					cluster_mix_dict[f'{cnst.MIX_DF_KEY}{i}'] = ', '.join(mixture_dict[cnst.REF_NAME_KEY])
					cluster_mix_dict[f'{cnst.CS_MIX_DF_KEY}{i}'] = float(f"{mixture_dict[cnst.CONF_SCORE_KEY]:.2f}")
					cluster_mix_dict[f'{cnst.MIX_MOLAR_RATIO_DF_KEY}{i}'] = np.round(mixture_dict[cnst.MOLAR_FR_MEAN_KEY] / (1 - mixture_dict[cnst.MOLAR_FR_MEAN_KEY]), 2)
					cluster_mix_dict[f'{cnst.MIX_FIRST_COMP_MEAN_DF_KEY}{i}'] = np.round(mixture_dict[cnst.MOLAR_FR_MEAN_KEY], 2)
					cluster_mix_dict[f'{cnst.MIX_FIRST_COMP_STDEV_DF_KEY}{i}'] = np.round(mixture_dict[cnst.MOLAR_FR_STDEV_KEY], 2)
				mixtures_strings_dict.append(cluster_mix_dict)
			else:
				mixtures_strings_dict.append({})

		mixtures_df = pd.DataFrame(mixtures_strings_dict)
		return mixtures_df
