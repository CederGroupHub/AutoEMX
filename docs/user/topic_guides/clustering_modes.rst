.. _clustering_modes:

Clustering modes
================

AutoEMX groups the measured compositions into clusters, ideally one per phase.
Two settings control this:

- ``method``: the clustering algorithm, ``'kmeans'`` (default) or ``'dbscan'``.
- ``geometry``: how the distance between two compositions is measured:
  ``'euclidean'``, ``'aitchison'``, or ``'auto'`` (chosen from the data; default
  for new samples).

The geometry only decides **which spectra are grouped together**. Reported
cluster compositions are always averages of the element fractions, so results
from different settings can be compared directly.

.. contents:: On this page
   :local:
   :depth: 1


Algorithms
----------

**k-means** assigns every composition to one of ``k`` clusters. ``k`` is either
forced (``k_forced``) or found automatically: AutoEMX first checks whether the
sample looks single-phase, and otherwise picks the best ``k`` up to ``max_k``
(by silhouette score by default), preferring fewer clusters when several values
are equally good.

**DBSCAN** finds dense groups of compositions and labels isolated ones as
*noise*. The number of clusters follows from the data, so ``k_forced`` is
ignored. Main parameters:

- ``eps``: how close two compositions must be to count as neighbors. If left
  unset, it defaults to **0.05** in Euclidean geometry and **0.3** in Aitchison
  geometry, since distances have very different scales in the two.
- ``min_samples``: minimum number of neighbors to form a cluster (default 3).

Use **k-means** in most cases. Try **DBSCAN** to pick out a few compact phases
from scattered compositions.


Geometries
----------

Euclidean
^^^^^^^^^

Distances are computed directly on the element fractions: a 1 at% difference
counts the same everywhere. It is simple and robust, but:

- minor elements barely matter: 1 at% vs 3 at% of an element is a small
  difference, even though it is a factor of 3;
- spectra with the same element ratios but a different oxygen content (e.g.
  from quantification scatter) are pulled apart.

Aitchison
^^^^^^^^^

Distances are computed on the **log-ratios** between elements (centered
log-ratio transform). Only ratios matter:

- 1 → 3 at% weighs as much as 10 → 30 at%;
- spectra with the same element ratios are identical, regardless of dilution or
  normalization (atomic and weight fractions give the same clusters).

This usually separates phases better when they **share the same elements in
different ratios**. Its weakness is that logarithms magnify small values, so
near-zero measurements (mostly noise in EDS) can dominate. The next section
explains how AutoEMX limits this.


Zeros and trace elements (Aitchison)
------------------------------------

The logarithm of zero is undefined, and the logarithm of a tiny, noisy value is
large and unstable. Before computing log-ratios, AutoEMX adjusts low values
using a detection limit, ``detection_limit_percent`` (default **0.5%**):

1. Elements that are zero in every spectrum (e.g. undetectable Li) are ignored.
2. Each element is classified from its **median** across the spectra: **trace**
   if below 1% (2 x the detection limit), **major** otherwise.
3. Low values are replaced with 0.325% (0.65 x the detection limit):

   ======= ============ =================== ================
   Element Exact zeros  Below 0.5%          Above 0.5%
   ======= ============ =================== ================
   Trace   replaced     replaced            kept
   Major   replaced     kept                kept
   ======= ============ =================== ================

Why:

- **Trace values are floored** because they are mostly noise. Without this, an
  impurity at 0.3 at% in some spectra and 0 in others would look like two
  different phases.
- **Low values of major elements are kept** because they are usually real, e.g.
  the element-poor end of a mixture.
- **Zeros are always replaced** because a measured zero means "below
  detection", not "absent".
- **The median is used, not the mean**, because it reflects a typical spectrum.
  An element present at 20 at% in only 10% of the spectra has a mean of about
  2% but a median of about 0. It is correctly treated as trace, and its 20 at%
  values are untouched by the floor.


How ``'auto'`` chooses
----------------------

With ``geometry='auto'``, AutoEMX uses **Euclidean** geometry if either of these
holds for the compositions being clustered, and **Aitchison** otherwise:

1. **At most two elements** are present. Log-ratios then reduce to a single
   number, which tends to split a single phase into many clusters.
2. **In at least 10% of spectra, a major element is below 1%.** This happens
   when phases have largely different elements, or along a mixture line that
   reaches an end-member. Log-ratios would magnify the noise in those
   near-zero values.

The choice is logged, for example::

   ℹ️ Automatic geometry selection: euclidean (a major element is < 1% in 61% of spectra (threshold 10%)).

The thresholds (``auto_near_zero_percent = 1.0``,
``auto_max_near_zero_fraction = 0.10``) were calibrated on a limited set of
samples. If the choice looks wrong for a sample, set ``geometry`` explicitly.

**Defaults.** New samples use ``'auto'``. Samples analyzed before this option
existed, and samples created from legacy data, use ``'euclidean'``, so
re-analysing them reproduces earlier results.


Examples
--------

In ``autoemx/scripts/Run_Analysis.py`` (options left as ``None`` keep the values
saved for the sample):

.. code-block:: python

   # Let AutoEMX choose the geometry (default for new samples)
   clustering_geometry = 'auto'

   # Force Aitchison geometry, flooring noisy trace elements more aggressively
   clustering_geometry = 'aitchison'
   aitchison_params = {'detection_limit_percent': 1.0}

   # DBSCAN, with eps defaulting to the chosen geometry
   clustering_method = 'dbscan'
   dbscan_params = {'min_samples': 5}

The same options are available in
:func:`analyze_sample <autoemx.runners.analyze_sample.analyze_sample>`
(``clustering_method``, ``clustering_geometry``, ``dbscan_params``,
``aitchison_params``).


Which setting to use
--------------------

.. list-table::
   :header-rows: 1

   * - Situation
     - Suggested setting
   * - Not sure
     - ``'auto'``
   * - Phases with the same elements in different ratios
     - ``'aitchison'``
   * - Phases with largely different elements, or 2-element samples
     - ``'euclidean'``
   * - Noisy trace elements splitting clusters (Aitchison)
     - raise ``detection_limit_percent``
   * - Comparing with older results
     - ``'euclidean'``

Always check the clustering plot: clusters should match visible groups of
compositions, with centroids near candidate phases or on the lines between them
(mixtures).
