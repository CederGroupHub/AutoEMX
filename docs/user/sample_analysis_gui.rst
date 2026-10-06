.. _sample_analysis_gui_tutorial:

Tutorial: Analyse AutoEMX samples in the interactive GUI
========================================================

The sample-analysis GUI runs the same quantification and clustering analysis as
``Run_Quantification.py`` and ``Run_Analysis.py`` on samples acquired with AutoEMX,
and lets you explore the results in interactive 3D, ternary and 2D plots.

It runs **locally**: a small server on your computer (``127.0.0.1``) that shows the
interface in your browser. It reads and writes the sample folders on your disk, and
nothing is uploaded anywhere.

Launch
------

::

   python -m autoemx.gui path/to/Results

The interface opens in your browser (http://127.0.0.1:8050). The folder can also be
chosen in the interface with **Browse…**. Options: ``--port 8051`` to use another port,
``--no-browser`` to not open a browser tab.

**Double-click launcher.** To open the GUI without a terminal, create a launcher once::

   python -m autoemx.gui --create-launcher

This writes ``AutoEMX GUI.command`` (macOS), ``AutoEMX GUI.bat`` (Windows) or
``AutoEMX GUI.sh`` (Linux) on your Desktop, which runs the GUI with the Python
environment used to create it. Add a results folder to open it directly
(``python -m autoemx.gui path/to/Results --create-launcher``) or a destination folder after
``--create-launcher``. Double-click it to open the GUI; the window that opens shows its log,
and closing it stops the GUI. Double-clicking again while the GUI runs just reopens it in
the browser.

Any folder containing sample folders (folders with a ``ledger.json``) can be used;
sub-folders are searched as well.

Workflow
--------

1. **Select a sample.** The sidebar shows its elements, substrate, number of spectra
   and of quantified spectra. The latest (active) analysis is shown.
2. **Set the parameters.** The form contains every quantification and clustering
   parameter, filled in with the values saved in the sample's ledger. Hover the ``?``
   icons for a description. **Load settings of shown analysis** fills the form with the
   settings of the analysis selected above the plot.

   - *Quantification*: only used when **Quantify spectra first** is ticked.
   - *Spectra filtering*: maximum analytical error and accepted quant flags
     (see :ref:`quant_flags`).
   - *Clustering*: method (k-means or DBSCAN), geometry, features, number of clusters
     (empty = found automatically), candidate phases, mixture decomposition.
   - *DBSCAN*, *Aitchison geometry*, *Cluster merging*, *Mixture decomposition*: all
     parameters of the corresponding configuration models.
   - *Plot output*: options of the PNG figures saved in the analysis folder.

3. **Run analysis.** The analysis runs in the background (a few seconds); its log is
   shown below the button and it can be cancelled. Tick **Quantify spectra first** to
   (re-)quantify the spectra before the analysis. Quantification takes about one
   minute per spectrum, so only re-quantify when needed. Results are saved to the
   ledger and to a new ``analysis_quant<i>_clust<j>`` folder, exactly as with the scripts.
4. **Explore the results.**

   - *Clustering*: 3D, ternary (3 elements, normalised) or 2D plot of the
     compositions, with clusters, ±1σ ellipsoids, centroids, candidate phases and the
     best mixture of each cluster. Choose the elements on each axis, colour the spectra
     by cluster, analytical error, particle, quant flag or R², and toggle each layer.
     Rotate and zoom with the mouse; click legend entries to hide them.
     **Save plot as HTML** saves the interactive plot in the analysis folder.
   - *Clusters*: composition ± standard deviation of each cluster, candidate phases
     and mixtures with their confidence and molar fractions.
   - *Spectra*: table of all spectra, sortable and filterable.
   - *Distributions*: fraction of each element by cluster.
   - *SEM images* (named after the microscope type saved in the ledger): the image of the
     particle (or frame) where the selected spectrum was collected, with all spectrum spots
     marked and coloured by cluster (click a spot to select its spectrum). ◀ / ▶ step
     through the images, which are also shown in a horizontal strip below, sorted by
     particle number.
   - *Saved figures* and *Settings used*: the PNG figures and
     ``Analysis_config_summary.txt`` of the analysis.

5. **Inspect single spectra.** Click a point in the plot, a row of the table or a point
   of the distributions to show the spectrum and its composition on the right; ◀ / ▶
   browse through the spectra. **Show in SEM image** opens the image where it was
   collected, with its spot highlighted. **Fit spectrum** re-fits and quantifies it (about 10 s)
   to show the fitted model, the background and the fitted peaks.

Previous analyses of a sample can be selected in the **Analysis** menu above the plot.
