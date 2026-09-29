.. _number_of_clusters:

Number of clusters
==================

With k-means clustering, the number of clusters ``k`` is either set by you or
chosen automatically. This page explains how the automatic choice works and how
to override it. (DBSCAN does not use ``k``: the number of clusters follows from
the data density, see :ref:`clustering_modes`.)

The automatic choice has three steps:

1. **Single-cluster check**: is the sample a single phase?
2. **Choosing k**: if not, how many clusters (between 2 and ``max_k``)?
3. **Merging clusters**: are some of the clusters pieces of one continuous
   population, e.g. a mixture line cut by k-means?

.. admonition:: Defaults: change them only if unsatisfied with the results

   By default ``k`` is **chosen automatically** (``k_forced = None``), using the
   silhouette score (``k_finding_method = 'silhouette'``) with up to
   ``max_k = 6`` clusters, then merges clusters that form one continuous
   population (``auto_merge_clusters = True``). The single-cluster threshold adapts
   to the feature type (see below).

   Set ``k`` yourself only if the result is unsatisfactory, e.g. a single phase
   split into several clusters, or a known phase missing (see `Setting k
   yourself`_).

.. contents:: On this page
   :local:
   :depth: 1


Step 1: single-cluster check
----------------------------

AutoEMX compares the data clustered into one group (``k = 1``) with the data
clustered into two groups (``k = 2``), using three numbers:

- **RMS distance**: how far spectra typically are from the average composition.
- **Silhouette score** of the best 2-cluster split (0 to 1): how clearly the two
  groups are separated. High means two distinct groups.
- **Inertia ratio**: how much the spread shrinks when splitting in two.

The rules are applied in this order; the first one that matches decides:

======================================= ================
Condition                               Result
======================================= ================
RMS distance below the threshold        single phase
Silhouette below 0.5                    single phase
Silhouette above 0.6                    multiple phases
Inertia ratio below 1.5                 single phase
Otherwise                               multiple phases
======================================= ================

A compact group of a few spectra that is clearly separated from the rest gives a
high silhouette, and is therefore reported as its own cluster, e.g. a minor
secondary phase or an impurity. This is intended.

The check always runs on the element fractions, whatever the clustering
geometry (Euclidean or Aitchison).

The RMS threshold
^^^^^^^^^^^^^^^^^

With **atomic fractions** (``features='at_fr'``), the threshold is **3%**.

With **weight fractions** (``features='w_fr'``), the fractions are not
normalized: each spectrum's total differs from 100% by its analytical error.
This adds spread that has nothing to do with the composition, so the threshold
is raised to allow for it:

.. math::

   \text{threshold} = \sqrt{0.03^2 + \sigma_\text{total}^2}

where :math:`\sigma_\text{total}` is the **standard deviation of the analytical
errors** of the spectra being clustered, as a fraction (e.g. 0.046 for 4.6
percentage points). The two sources of spread are independent, so they add in
quadrature. The noisier the totals, the more spread is accepted before calling
the sample multi-phase.

:math:`\sigma_\text{total}` is capped at the **maximum analytical error** used to
filter spectra (``max_analytical_error_percent``, or 10% if no filter is set).
Spectra within the filter window cannot scatter more than that. Larger scatter
is possible only through the extra allowance for undetectable elements (e.g.
Li), and then likely reflects real differences between phases rather than noise.

With atomic fractions every total is 100%, so :math:`\sigma_\text{total} = 0`
and the formula gives 3%.

.. note::

   Use weight fractions when only **two elements** are quantified (e.g. oxides
   of a single metal, or Li-containing samples where Li is not detected). With
   atomic fractions all spectra fall on a line. The analytical total adds a
   second dimension, and carries information about undetected elements.


Step 2: choosing k
------------------

If the sample is not a single phase, ``k`` is chosen between 2 and ``max_k``
(default 6) using the ``k_finding_method``:

- ``'silhouette'`` (default): the ``k`` with the best silhouette score, i.e.
  the most clearly separated clusters.
- ``'calinski_harabasz'``: the ``k`` with the best ratio of between-cluster to
  within-cluster spread.
- ``'elbow'``: the ``k`` after which adding clusters stops reducing the spread
  much.

Because k-means starts from random positions, the search is repeated 20 times
and the most frequent ``k`` is taken, provided it is at least twice as frequent
as the next one. Otherwise the search is repeated, up to 5 rounds. If there is
still no clear winner, the **smallest** ``k`` that is at least half as frequent
as the top one is chosen. When several values are about equally good, fewer
clusters are preferred.

In Aitchison geometry, this search runs on the log-ratio coordinates, the same
space the clustering uses.

Once ``k`` is chosen, k-means is run several times and the solution with the
best silhouette score is kept.


Step 3: merging clusters
------------------------

k-means divides the compositions into compact groups, even when they spread
continuously. A mixture of two phases measured at many spots spreads along the
line between them, and k-means often cuts it into several clusters although it
is one population. With ``auto_merge_clusters = True`` (default), the clusters
are checked in pairs after k-means, and merged when all of the following hold:

1. **One mode.** Along the axis joining the two cluster centers, their
   compositions form a single peak: Hartigan's dip test gives a p-value above
   0.05. An even spread along a mixture line counts as one peak.
2. **No gap.** The empty space between the two clusters along that axis is
   smaller than the spread (standard deviation) of the larger cluster. This
   keeps small groups clear of the main one, e.g. a few spots of a minor phase,
   which the dip test alone cannot detect.
3. **Connected.** DBSCAN links the two clusters through dense neighbourhoods
   of compositions. This keeps apart groups that touch in one direction but
   differ in shape, e.g. a compact phase next to a diffuse cloud. The
   neighbourhood size adapts to each sample: 3 times the median distance of the
   compositions to their 5th nearest neighbour.

Pairs are merged one at a time, starting from the most compatible, and the
checks are repeated after each merge. Clusters are never split. The checks use
the clustering geometry (log-ratio coordinates in Aitchison geometry).

The merge only runs when ``k`` is chosen automatically: a forced ``k`` is always
kept. The merge is deliberately conservative: when in doubt, clusters stay apart.

A merged cluster that spreads along a mixture line is then described by the
:ref:`mixture decomposition <mixture_decomposition>`, which reports its phases
and their fractions.

To disable the merge, set ``auto_merge_clusters = False`` in
:func:`analyze_sample <autoemx.runners.analyze_sample.analyze_sample>`. Samples
analysed before this option existed keep their previous results
(``auto_merge_clusters = False``). The thresholds are in ``ClusterMergeParams``
(``ClusteringConfig.cluster_merge``): ``dip_alpha`` (0.05), ``max_gap_ratio``
(1.0), ``dbscan_min_samples`` (5) and ``dbscan_eps_factor`` (3.0).


Setting k yourself
------------------

- ``k_forced = 3``: always use 3 clusters (no automatic choice).
- ``k_forced = None``: use the settings saved for the sample. If ``k`` was found
  automatically before, it is found again on each run.
- ``k_forced = False`` (in :func:`analyze_sample <autoemx.runners.analyze_sample.analyze_sample>`):
  discard a previously forced ``k`` and find it automatically.
- ``k_finding_method``: change the method used in step 2. Setting it also
  forces ``k`` to be re-evaluated.
- ``max_k``: largest ``k`` considered in step 2.
- ``auto_merge_clusters``: whether step 3 runs (default ``True``).

Example, in ``autoemx/scripts/Run_Analysis.py``:

.. code-block:: python

   k_forced = None                   # choose k automatically
   k_finding_method = 'silhouette'   # or 'calinski_harabasz', 'elbow'

   # or, to force it:
   k_forced = 2


Reading the result
------------------

With verbose output, the log shows the single-cluster check, for example::

   📊 RMS distance for k=1: 5.1% (single-cluster threshold: 5.5%)
   📊 Silhouette Score for k=2: 0.51
   ℹ️ d_rms < 5.5%: The data effectively forms a single cluster.

When more than one cluster is found and plots are saved, ``Silhouette_plot.png``
shows how well each spectrum fits its cluster.

When to override the automatic choice:

- **A single phase is split into two or more clusters** without a clear gap in
  the clustering plot: set ``k_forced = 1``, or check whether ``w_fr`` should be
  used (two-element samples).
- **A known minor phase is merged** into the main cluster: set ``k_forced`` to
  the expected number of phases.
- **A continuous spread (e.g. a mixture line) is still cut into several clusters**
  after merging: force a smaller ``k``. Mixtures are better described by the
  mixture decomposition than by many clusters.
- **Two phases that should stay apart are merged**: set ``auto_merge_clusters = False``
  or force ``k``.
