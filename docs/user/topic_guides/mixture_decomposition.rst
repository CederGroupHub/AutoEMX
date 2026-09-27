.. _mixture_decomposition:

Mixture decomposition
=====================

A cluster can come from a single phase, or from spots that probe several phases
at once: fine intermixed powders, unreacted precursors, or partly reacted samples.
AutoEMX then describes the cluster as a **mixture of candidate phases**
(``ref_formulae``), and reports which phases and in which proportions. This page
explains how the phases are chosen, how many, and what is reported.

.. admonition:: Defaults: change them only if unsatisfied with the results

   The mixture decomposition runs by default (``do_matrix_decomposition = True``)
   and needs no settings besides the candidate phases, ``ref_formulae``. Its
   defaults (``MixtureParams``) were tested on single-phase standards, known binary
   mixtures and multi-phase sol-gel samples, and are not exposed in
   ``Run_Analysis.py``.

   Change them only if the results are unsatisfactory, e.g. a mixture you know is
   binary is reported with three phases (see `Changing the defaults`_).

.. contents:: On this page
   :local:
   :depth: 1


The idea: the shape of the cluster
----------------------------------

Each spot measures a mix of phases, so its composition is a weighted average of
the phase compositions. The spots of a cluster therefore fill the region between
its phases:

- a **binary** mixture spreads along the **line** between the two phases, with a
  width set by the measurement noise;
- a **ternary** mixture spreads over the **plane** (triangle) between three phases.

.. figure:: /_static/Example_mixture_plot_binary.png
   :alt: Binary mixture in a ternary diagram
   :width: 45%
   :align: center

   A binary mixture (RPA1_24, CaCO₃ + 2 Ta₂O₅): spots along the Ta₂O₅–Ca₄Ta₂O₉ line.

Adding phases always reduces the reconstruction error, even when the extra phase
is not there. So AutoEMX uses **the fewest phases that reproduce the cluster within
noise**: it tests pairs of candidate phases first, and only tries three (then four)
phases if no pair is good enough.


Which clusters are decomposed
-----------------------------

A cluster is considered a **single phase**, and is not decomposed, if its spots are
compact (RMS distance from the centroid below 3%) and it matches a candidate phase
(confidence above 0.5). With ``is_known_powder_mixture_meas = True`` (known precursor
mixtures, see :ref:`precursor_mix_tutorial`), every cluster is decomposed.


How a combination of phases is scored
-------------------------------------

For a combination of candidate phases, each spot is written as the non-negative
combination of the phase compositions that best reproduces it (weights summing to 1).
The mismatch over all spots is the **reconstruction error**, an exponential penalty
that weighs large deviations much more than small ones:

.. math::

   e = \operatorname{mean}\left(e^{15\,|X - WH|} - 1\right)

where :math:`X` are the measured fractions, :math:`H` the phase compositions and
:math:`W` the weights. The error is turned into a **confidence score**:

.. math::

   \mathrm{CS_{mix}} = \exp\left(-\frac{e^2}{2 \cdot 0.5^2}\right)

.. list-table::
   :header-rows: 1

   * - Reconstruction error :math:`e`
     - :math:`\mathrm{CS_{mix}}`
     - Typical case
   * - 0.1 – 0.3
     - 0.84 – 0.98
     - binary mixture, correct pair
   * - 0.4
     - 0.73
     - limit for "explains the cluster"
   * - 0.7 – 2
     - 0.35 – 0.0004
     - a third phase is missing

The weights are converted into **molar fractions** of the phases (dividing by the
number of atoms, or the mass, per formula unit of each phase).


How many phases
---------------

1. **Pairs.** All pairs of candidate phases are fitted. Pairs with :math:`e < 1`
   (:math:`\mathrm{CS_{mix}} \geq 0.14`) are kept, for inspection.
2. **Three phases**, only if no pair has :math:`e < 0.4`: all combinations of three
   candidate phases are fitted, and those with :math:`e < 0.4` are kept.
3. **Four phases**, only if no combination of three has :math:`e < 0.4`.

The search stops at the first number of phases that reaches :math:`e < 0.4`.
The limit 0.4 separates the three groups of test samples:

============================ ====================== ======================
Samples                      Best fit with          Error
============================ ====================== ======================
Single-phase standards (21)  1 phase                ≤ 0.28
Known binary mixtures (7)    1 phase / 2 phases     ≥ 0.47 / ≤ 0.30
Sol-gel samples (19)         2 phases / 3 phases    ≥ 0.73 / ≤ 0.05
============================ ====================== ======================

.. note::

   With three detectable elements, three phases whose compositions surround the
   cluster always reproduce it. The decision to use three phases is then reliable
   (the pairs failed), but which three phases are chosen depends on the candidate
   list. Candidates close to a pure element (e.g. ``CO``, which is pure O when C is
   not quantified) can enclose almost any composition.

If no candidate phase combination explains the cluster (best
:math:`\mathrm{CS_{mix}} < 0.5`), AutoEMX also fits **two phases of unknown
composition** (free NMF), and reports them with approximate formulas.


Equivalent combinations
-----------------------

Phases on the same line give identical fits. For example, in a CaO–Ta₂O₅ sample,
CaO + Ta₂O₅ and Ca₄Ta₂O₉ + Ta₂O₅ reproduce the spots equally well, because Ca₄Ta₂O₉
lies on the CaO–Ta₂O₅ line. The lowest error alone would pick one of them
arbitrarily. Instead, combinations within 0.01 of the best error are ranked:

1. fewer phases first;
2. then the phases **closest to each other** in composition space (smallest sum
   of distances between the phases), i.e. the pair that brackets the spots most
   tightly: here Ca₄Ta₂O₉ + Ta₂O₅.


What is reported
----------------

All mixtures found are saved in ``ledger.json``, with their rank, confidence,
reconstruction error and molar fractions. ``Clusters.csv`` shows a selection:

- **Equivalent decompositions are merged.** A mixture whose phases span the same
  line or plane as a better-ranked one describes the same mixture (e.g. any pair
  of Sr–Ta oxides, which all lie on the SrO–Ta₂O₅ line) and is not shown. Three
  phases spanning the whole composition space are always distinct.
- **At most 5 mixtures** are shown, plus any within 10% of the best confidence.
- **None below 50%** of the best confidence.

The ``Mix_more`` row gives the number of mixtures only saved in the ledger, and
why, e.g.::

   18 more mixture(s) saved in ledger.json (clusters_assigned_mixtures of this
   clustering analysis): 16 equivalent to a listed mixture (same mixing
   line/plane); 2 with confidence below 50% of the best (CS_mix < 0.47)

To read all mixtures of a cluster:

.. code-block:: python

   from autoemx.config import load_sample_ledger

   ledger = load_sample_ledger("path/to/sample/ledger.json")
   analysis = ledger.quantifications[ledger.active_quant].get_active_clustering_analysis()
   mixtures = analysis.result.clusters_assigned_mixtures[cluster_id]
   for mix in sorted(mixtures, key=lambda m: m.get("rank", 0)):
       print(mix["refs"], mix["conf_score"], mix.get("means"))

The columns of each mixture in ``Clusters.csv`` are described in
:doc:`../comp_analysis` (``Mix``, ``CS_mix``, ``Mol_Ratio``, ``X1_mean``, ``X_means``).


Plots
-----

``Mixture_plot_cl<i>.png`` (and ``_zoomed``) shows the top-ranked mixture of
cluster ``i``, when it has 2 or 3 candidate phases (``PlotConfig.plot_best_mixture``,
default ``True``):

- **3 detectable elements**: ternary diagram of the elements, with the mixture
  phases (line or triangle), the other candidate phases and, for 2 phases, each
  spot's position on the mixture line.
- **2 phases, other numbers of elements**: the same in 2D, using the two elements
  that differ most between the phases (or ``els_to_plot``, if it lists two).
- **3 phases, 4 or more elements**: ternary diagram of the molar fractions of the
  three phases, with spots coloured by their distance from the mixture plane.

For known powder mixtures (``is_known_powder_mixture_meas = True``), a violin plot
of the molar fractions is also drawn for each mixture shown in ``Clusters.csv``.


Changing the defaults
---------------------

The options are in ``MixtureParams`` (``ClusteringConfig.mixture``). Set them only
if needed, through :func:`analyze_sample <autoemx.runners.analyze_sample.analyze_sample>`:

.. code-block:: python

   analyze_sample(sample_ID, mixture_params={"max_n_phases": 2})   # binary mixtures only

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Option
     - Default
     - Meaning
   * - ``max_n_phases``
     - 4
     - Most phases combined in a mixture. ``2``: binary mixtures only.
   * - ``max_recon_error``
     - 0.4
     - Error below which a combination explains the cluster (and more phases are not tried).
   * - ``max_recon_error_binary``
     - 1
     - Pairs with a lower error are kept, for inspection.
   * - ``equivalent_recon_error_tol``
     - 0.01
     - Combinations within this error of the best are ranked by number of phases, then by spread.
   * - ``max_reported_mixtures``, ``report_within_conf_ratio``, ``min_reported_conf_ratio``
     - 5, 0.9, 0.5
     - Which mixtures appear in ``Clusters.csv`` (see `What is reported`_).
   * - ``collapse_equivalent_mixtures``, ``equivalent_span_tol``
     - True, 0.005
     - Merge equivalent decompositions in ``Clusters.csv``.
   * - ``recon_error_alpha``, ``conf_sigma``
     - 15, 0.5
     - Reconstruction error and confidence score definitions.
   * - ``single_phase_max_rms_dist``, ``single_phase_min_ref_conf``
     - 0.03, 0.5
     - When a cluster is a single phase and is not decomposed.
   * - ``nmf_min_mixture_conf``
     - 0.5
     - Below this confidence, two phases of unknown composition are also fitted.

When to change them:

- **A known binary is reported with three phases**: its spots scatter more than
  usual (e.g. rough particles). Check the mixture plot; if the third phase is not
  plausible, set ``max_n_phases = 2``.
- **A known third phase is missing**: make sure it is among ``ref_formulae``. A
  phase that is not a candidate cannot be found.


Limitations
-----------

- Only candidate phases can be found; the free-NMF fallback is limited to two phases.
- Phases lying on one line in composition space (e.g. CuO, CuAl₂O₄ and Al₂O₃)
  cannot be told apart by composition alone: they are reported as equivalent
  decompositions of the same binary mixture.
- Elements that are not quantified (e.g. H, Li, and C when not in the sample
  elements) are ignored in the phase compositions: ``LiNb₂O₅`` and ``Nb₂O₅`` would look the same.
