#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting mixin for composition analysis outputs."""

import os
import warnings
from typing import List

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer

import autoemxsp.XSp_calibs as calibs
import autoemxsp._custom_plotting as custom_plotting
import autoemxsp.utils.constants as cnst
from autoemxsp.utils import print_single_separator, to_latex_formula


class PlottingModule:
    def _save_plots(
        self,
        kmeans: 'KMeans',
        compositions_df: 'pd.DataFrame',
        centroids: 'np.ndarray',
        labels: 'np.ndarray',
        els_std_dev_per_cluster: list,
        unused_compositions_list: list
    ) -> None:
        # Silhouette plot (only if more than one cluster)
        if len(centroids) > 1:
            PlottingModule._save_silhouette_plot(
                kmeans, compositions_df, self.analysis_dir, show_plot=self.plot_cfg.show_plots
            )

        can_plot_clustering = True
        els_for_plot = list(set(self.detectable_els_sample) - set(self.plot_cfg.els_excluded_clust_plot))
        els_excluded_clust_plot = list(set(self.all_els_sample) - set(els_for_plot))
        n_els = len(els_for_plot)

        if n_els == 1:
            can_plot_clustering = False
            print_single_separator()
            warnings.warn("Cannot generate clustering plot with a single element.", UserWarning)
        elif n_els > 3:
            els_excluded_clust_plot += els_for_plot[3:]
            els_for_plot = els_for_plot[:3]

        indices_to_remove = [self.all_els_sample.index(el) for el in els_excluded_clust_plot]
        els_for_plot = [el for i, el in enumerate(self.all_els_sample) if i not in indices_to_remove]
        centroids = np.array([[coord for i, coord in enumerate(row) if i not in indices_to_remove] for row in centroids])
        els_std_dev_per_cluster = [[stddev for i, stddev in enumerate(row) if i not in indices_to_remove] for row in els_std_dev_per_cluster]
        unused_compositions_list = [[fr for i, fr in enumerate(row) if i not in indices_to_remove] for row in unused_compositions_list]

        if can_plot_clustering:
            els_comps_list = compositions_df[els_for_plot].to_numpy().T
            if self.plot_cfg.use_custom_plots:
                custom_plotting._save_clustering_plot_custom_3D(
                    els_for_plot, els_comps_list, centroids, labels,
                    els_std_dev_per_cluster, unused_compositions_list,
                    self.clustering_cfg.features, self.ref_phases_df,
                    self.ref_formulae, self.plot_cfg.show_plots, self.sample_cfg.ID
                )
            else:
                self._save_clustering_plot(
                    els_for_plot, els_comps_list, centroids, labels,
                    els_std_dev_per_cluster, unused_compositions_list
                )
        elif self.verbose:
            print('Clusters were not plotted because only one detectable element was present in the sample.')
            print(f"Elements {calibs.undetectable_els} cannot be detected at the employed instrument.")

    def _save_clustering_plot(
        self,
        elements: List[str],
        els_comps_list: 'np.ndarray',
        centroids: 'np.ndarray',
        labels: 'np.ndarray',
        els_std_dev_per_cluster: list,
        unused_compositions_list: list
    ) -> None:
        plt.rcParams['font.family'] = 'Arial'
        fontsize = 14
        labelpad = 12
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize

        axis_label_add = ' (w%)' if self.clustering_cfg.features == cnst.W_FR_CL_FEAT else ' (at%)'
        ticks = np.arange(0, 1, 0.1)
        ticks_labels = [f"{x*100:.0f}" for x in ticks]

        fig = plt.figure(figsize=(6, 6))
        if len(elements) == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlabel(elements[2] + axis_label_add, labelpad=labelpad * 0.95)
            ax.set_zlim(0, 1)
            ax.set_zticks(ticks)
            ax.set_zticklabels(ticks_labels)
        else:
            ax = fig.add_subplot(111)

        ax.scatter(*els_comps_list, c=labels, cmap='viridis', marker='o')
        ax.scatter(*centroids.T, c='red', marker='x', s=100, label='Centroids')

        first_ellipse = True
        for centroid, stdevs in zip(centroids, els_std_dev_per_cluster):
            if ~np.any(np.isnan(stdevs)):
                if len(elements) == 3:
                    x_c, y_c, z_c = centroid
                    rx, ry, rz = stdevs
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    x = x_c + rx * np.outer(np.cos(u), np.sin(v))
                    y = y_c + ry * np.outer(np.sin(u), np.sin(v))
                    z = z_c + rz * np.outer(np.ones_like(u), np.cos(v))
                    ax.plot_surface(x, y, z, color='red', alpha=0.1, edgecolor='none')
                    if first_ellipse:
                        first_ellipse = False
                        ax.plot([], [], [], color='red', alpha=0.1, label='Stddev')
                else:
                    x_c, y_c = centroid
                    rx, ry = stdevs
                    ellipse = patches.Ellipse((x_c, y_c), rx, ry, edgecolor='red', facecolor='red', linestyle='--', alpha=0.2)
                    if first_ellipse:
                        ellipse.set_label('Stddev')
                        first_ellipse = False
                    ax.add_patch(ellipse)

        if unused_compositions_list and self.plot_cfg.show_unused_comps_clust:
            ax.scatter(*np.array(unused_compositions_list).T, c='grey', marker='^', label='Discarded comps.')

        if self.ref_formulae is not None:
            first_ref = True
            ref_phases_df = self.ref_phases_df[elements]
            for index, row in ref_phases_df.iterrows():
                label = 'Candidate phases' if first_ref else None
                ax.scatter(*row.values, c='blue', marker='*', s=100, label=label)
                ref_label = to_latex_formula(self.ref_formulae[index])
                ax.text(*row.values, ref_label, color='black', fontsize=fontsize, ha='left', va='bottom')
                first_ref = False

        for i, centroid in enumerate(centroids):
            ax.text(*centroid, str(i), color='black', fontsize=fontsize, ha='right', va='bottom')

        ax.set_xlabel(elements[0] + axis_label_add, labelpad=labelpad)
        ax.set_ylabel(elements[1] + axis_label_add, labelpad=labelpad)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks_labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks_labels)
        ax.set_title(f'{self.clustering_cfg.method} clustering {self.sample_cfg.ID}')

        if getattr(self.plot_cfg, 'show_legend_clustering', None):
            ax.legend(fontsize=fontsize)

        if self.plot_cfg.show_plots:
            plt.show()
        fig.savefig(os.path.join(self.analysis_dir, cnst.CLUSTERING_PLOT_FILENAME + cnst.CLUSTERING_PLOT_FILEEXT))

    def _save_violin_plot_powder_mixture(
        self,
        W_mol_frs: List[float],
        ref_names: List[str],
        cluster_ID: int
    ) -> None:
        plt.rcParams['font.family'] = 'Arial'
        fontsize = 17
        labelpad = 0
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
        purple_cmap = cm.get_cmap('Purples')
        yellow_cmap = cm.get_cmap('autumn')

        y_vals = W_mol_frs[:, 0]
        fig, ax_left = plt.subplots(figsize=(4, 4))
        mean = np.mean(y_vals)
        std = np.std(y_vals)

        ax_left = sns.violinplot(data=y_vals, inner=None, color=purple_cmap(0.3), linewidth=1.5, density_norm='area', width=1, zorder=1)
        sns.swarmplot(data=y_vals, color=purple_cmap(0.8), edgecolor=purple_cmap(1.0), linewidth=2, size=5, label='data', zorder=2)
        ax_left.errorbar(0, mean, yerr=std / 2, fmt='none', color=yellow_cmap(0.9), label='Mean ±1 Std Dev', capsize=5, elinewidth=1, zorder=4, markerfacecolor=yellow_cmap(0.9), markeredgecolor='black', markeredgewidth=1, marker='o', linestyle='none')
        ax_left.errorbar(0, mean, yerr=std / 2, fmt='none', color='none', label='_nolegend_', capsize=6, elinewidth=2, zorder=3, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, marker='o', linestyle='none', ecolor='black')
        ax_left.scatter(0, mean, color=yellow_cmap(0.9), marker='o', s=50, edgecolors='k', linewidths=1, label='Mean', zorder=10)

        ax_left.set_xticks([])
        ax_left.set_yticks([0, 1])
        ax_left.set_frame_on(True)
        for spine in ax_left.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.5)
        plt.grid(False)

        plt.xlim(-0.5, 0.5)
        ylim_bottom = 0
        ylim_top = 1
        ax_left.set_ylim(ylim_bottom, ylim_top)

        left_formula = to_latex_formula(ref_names[0], include_dollar_signs=False)
        ax_left.set_ylabel(rf"$x_{{\mathrm{{{left_formula}}}}}$", labelpad=labelpad)
        ax_right = ax_left.twinx()
        ax_right.set_ylim(ylim_top, ylim_bottom)
        ax_right.set_yticks([1, 0])
        right_formula = to_latex_formula(ref_names[1], include_dollar_signs=False)
        ax_right.set_ylabel(rf"$x_{{\mathrm{{{right_formula}}}}}$", labelpad=labelpad)
        ax_left.text(0.03, 0.03, rf"$\sigma_x = {std*100:.1f}$%", fontsize=fontsize, color='black', ha='left', va='bottom', transform=ax_left.transAxes)
        ax_left.set_title(f'Violin plot {self.sample_cfg.ID}')

        fig.savefig(
            os.path.join(self.analysis_dir, cnst.POWDER_MIXTURE_PLOT_FILENAME + f"_cl{cluster_ID}_{ref_names[0]}_{ref_names[1]}" + cnst.CLUSTERING_PLOT_FILEEXT),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
        )

    @staticmethod
    def _save_silhouette_plot(
        model: 'KMeans',
        compositions_df: 'pd.DataFrame',
        results_dir: str,
        show_plot: bool
    ) -> None:
        plt.figure(figsize=(10, 8))
        sil_visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sil_visualizer.fit(compositions_df)

        plt.ylabel('Cluster label')
        plt.xlabel('Silhouette coefficient values')
        plt.legend(loc='upper right', frameon=True)

        if show_plot:
            plt.ion()
            sil_visualizer.show()
            plt.pause(0.001)
            plt.ioff()

        fig = sil_visualizer.fig
        fig.savefig(os.path.join(results_dir, 'Silhouette_plot.png'))
        if not show_plot:
            plt.close(fig)
