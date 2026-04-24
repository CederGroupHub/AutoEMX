"""User-editable template for clustering custom plots.

This file is copied into each sample folder as custom_plot.py when
plot_custom_plots=True is used and no file exists yet.

Edit this file freely to produce publication-ready plots.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def _save_clustering_plot_custom_3D(
    elements,
    els_comps_list,
    centroids,
    labels,
    els_std_dev_per_cluster,
    unused_compositions_list,
    clustering_features,
    ref_phases_df,
    ref_formulae,
    show_plots,
    sample_ID,
    analysis_dir=None,
    output_filename=None,
):
    """Render and save a user-customized clustering plot.

    Keep this function name and required arguments so AutoEMXSp can call it.
    """
    axis_units = "(w%)" if str(clustering_features).startswith("w") else "(at%)"

    fig = plt.figure(figsize=(6, 6))
    if len(elements) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*els_comps_list, c=labels, cmap="viridis", s=36, marker="o", alpha=0.95)
        ax.scatter(*centroids.T, c="crimson", marker="x", s=90, label="Centroids")
        ax.set_zlabel(f"{elements[2]} {axis_units}")
        ax.set_zlim(0, 1)
        ax.view_init(elev=24, azim=40)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(*els_comps_list, c=labels, cmap="viridis", s=36, marker="o", alpha=0.95)
        ax.scatter(*centroids.T, c="crimson", marker="x", s=90, label="Centroids")

    if unused_compositions_list:
        ax.scatter(*np.array(unused_compositions_list).T, c="black", marker="^", s=30, label="Discarded comps.")

    ax.set_xlabel(f"{elements[0]} {axis_units}")
    ax.set_ylabel(f"{elements[1]} {axis_units}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Custom clustering plot {sample_ID}")
    ax.legend(loc="best")

    if show_plots:
        plt.ion()
        plt.show()
        plt.pause(0.001)

    filename = output_filename or "Clustering_plot_custom.png"
    if analysis_dir:
        plot_path = os.path.join(analysis_dir, filename)
    else:
        plot_path = filename

    fig.savefig(plot_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    if not show_plots:
        plt.close(fig)
