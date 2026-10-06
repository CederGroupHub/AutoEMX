.. _sample_analysis_gui_tutorial:

Tutorial: Quantify and analyse AutoEMX samples in the interactive GUI
=====================================================================

The AutoEMX GUI runs the same acquisition, quantification and clustering analysis as
``Run_Acquisition.py``, ``Run_Quantification.py``, ``Run_Analysis.py`` and
``Fit_Quant_Single_AutoEMX_Spectrum.py``, and lets you explore the results in interactive 3D,
ternary and 2D plots. It has four tabs: **Acquisition**, **Quantification**, **Analysis** and
**Single spectrum**. The results folder and the current sample, chosen at the top, are
shared by all tabs.

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
environment used to create it. On macOS the launcher shows the AutoEMX icon. Add a results folder to open it directly
(``python -m autoemx.gui path/to/Results --create-launcher``) or a destination folder after
``--create-launcher``. Double-click it to open the GUI; the window that opens shows its log,
and closing it stops the GUI. Double-clicking again while the GUI runs just reopens it in
the browser.

Any folder containing sample folders (folders with a ``ledger.json``) can be used;
sub-folders are searched as well.

Acquisition tab
---------------

Acquires the spectra of a list of samples with the electron microscope, like
``Run_Acquisition.py``. Use it on the microscope computer: the tab shows whether the microscope
API (e.g. PyPhenom for the Phenom XL) is available, and the GUI opens on this tab when it is.

1. **Samples to acquire.** One row per sample: ID (name of its folder in the results folder),
   elements to quantify, stage position of the sample centre (x, y in mm) and, optionally,
   candidate phases. **Add sample** adds a row, ⧉ copies a row
   (inserted below it, with the ID suffixed ``_copy``) and × deletes one. Acquiring a sample that already
   exists adds spectra to it.
2. **Settings.** Every option of ``Run_Acquisition.py``: sample & holder (sample type, substrate
   and substrate elements, automatic carbon-tape detection, working distance), acquisition (beam
   energy, target counts, maximum acquisition time, quantification during acquisition, number of
   spectra, manual navigation, brightness/contrast), images (format, annotation of particle images,
   raw copies), powder and bulk-grid acquisition parameters, and the quantification settings used
   when quantifying during acquisition. Without quantification during acquisition, a fixed
   **number of spectra** is collected; with it, spectra are collected until the clustering converges,
   between **min** and **max spectra**. Options that do not apply are greyed out (e.g. powder
   parameters for bulk samples) or hidden (contrast and brightness with automatic adjustment).
   **Reset to defaults** restores the defaults of the script.
3. **Start acquisition** asks for confirmation, then acquires the samples one after the other,
   with a progress bar per sample (spectra acquired) and the log. With manual navigation or
   manual particle selection, a window asks you to centre each spot or particle. **Stop**
   interrupts the acquisition, keeping the spectra already acquired; the microscope is left in
   its current state. When the acquisition ends, the results folder is rescanned and the new
   samples appear in the other tabs.

**Save as script…** downloads a Python script running the same acquisition, e.g. to run it on
the microscope computer without the GUI. The settings and the sample list are kept in the
browser between sessions.

Quantification tab
------------------

Quantifies the spectra of one or more samples, like ``Run_Quantification.py``.

1. **Tick the samples to quantify** in the table. All samples start unticked, so you can
   quantify only the samples just added to a project. Sort the table by clicking a column
   header, e.g. **Acquired** (date of the first spectrum) to show the latest samples first,
   and filter it by typing in the row below the headers. **Select/Unselect all** ticks (or
   unticks) all the samples left by the filter.
2. **Elements.** The sample and substrate elements of each sample can be edited in the table.
   Unchanged elements keep the values saved in the sample's ledger.
3. **Settings.** Options left empty (or set to *saved*) keep each sample's saved value. When the
   ticked samples share the same saved value, it is shown in grey in the empty field (and as
   e.g. *saved (no)* in the menus); *saved* means that their values differ. The ↺ button next to
   a field empties it, i.e. goes back to the saved value.
   *Which spectra*: interrupt fits of bad spectra, force a new quantification run,
   re-quantify only spectra without a composition, and **max spectra per sample** to quantify
   only the first N spectra, e.g. for a quick test. **Run analysis afterwards** (on by default)
   also runs the clustering analysis with each sample's saved settings.
4. **Quantify and analyse selected samples** (or **Quantify selected samples**). The samples are quantified one after the other, in the
   order shown in the table, with a progress bar per sample and the log. Quantification takes
   up to a minute per spectrum (spectra are processed in parallel). Cancelling keeps the spectra
   already quantified.

**Import spectra folder…** creates a sample from a folder of EMSA spectra (``.msa``, ``.emsa``,
``.msg``), e.g. acquired without AutoEMX, like ``quantify_external_spectra.py`` (see
:doc:`quantify_external_spectra`). Choose the folder: the number of spectra, and the beam energy
and energy calibration written in their headers, are shown. The energy calibration is always
taken from the headers (``#OFFSET`` and ``#XPERCHAN``); folders without it, or with spectra of
different calibrations, cannot be imported. Give the sample ID (by default the
folder name), the sample and substrate elements, the sample type, the microscope whose P/B
standards are used for the quantification, and the beam energy (filled in from the headers). **Import** copies the spectra into a new sample folder of the results
folder and writes its ledger, without modifying the original files. The new sample then appears in
the table, to be quantified and analysed like the others. Candidate phases are set in the Analysis
tab.

Click a sample in the table to show its quantification runs (settings, number of quantified
spectra, quant flags; see :ref:`quant_flags`) and the flags of its active run.
**Analyse this sample →** opens it in the Analysis tab.

The table reads the ledger of every sample; on cloud drives, ledgers not stored locally can
take a few seconds each to download the first time. Summaries are cached in
``~/.autoemx/gui_sample_summaries.json`` and only re-read when a ledger changes.

Analysis tab
------------

1. **Select a sample** at the top. The sidebar shows its elements, substrate, number of
   spectra and of quantified spectra. The latest (active) analysis is shown.
2. **Set the parameters.** The form contains every clustering parameter, filled in with the
   values saved in the sample's ledger. Parameters not used with the current settings are
   greyed out. Hover the ``?``
   icons for a description. **Load settings of shown analysis** fills the form with the
   settings of the analysis selected above the plot.

   - *Spectra filtering*: maximum analytical error and accepted quant flags
     (see :ref:`quant_flags`).
   - *Clustering*: method (k-means or DBSCAN), geometry, features, number of clusters
     (empty = found automatically), candidate phases, mixture decomposition.
   - *DBSCAN*, *Aitchison geometry*, *Cluster merging*, *Mixture decomposition*: all
     parameters of the corresponding configuration models.
   - *Plot output*: options of the PNG figures saved in the analysis folder.

3. **Run analysis.** The analysis of the active quantification runs in the background
   (a few seconds); its log is shown below the button and it can be cancelled. Results are
   saved to the ledger and to a new ``analysis_quant<i>_clust<j>`` folder, exactly as with
   the scripts.
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
   with the settings of the active quantification, to show the fitted model, the background and
   the background counts under the reference peaks. **Open in Single spectrum** opens it in the
   Single spectrum tab, to fit it with other settings.

Previous analyses of a sample can be selected in the **Analysis** menu above the plot.

Single spectrum tab
-------------------

Fits and quantifies one spectrum with every option of ``Fit_Quant_Single_AutoEMX_Spectrum.py``.
Results are shown only: they are not saved to the ledger.

1. **Choose the spectrum**: a spectrum of the current sample (menu, or ◀ / ▶), or an
   **external file** (``.msa``, ``.emsa``, ``.msg``) chosen with **Browse…** or dropped on the
   page. The geometry of external spectra (beam energy, emergence angle, live time) is read from
   their header.
2. **Settings**: sample and substrate elements, standard of known composition (with its formula),
   quantify or only fit, particle geometry, fit tolerance, spectrum limits, maximum undetectable mass fraction, single iteration, interrupt fit of bad spectra, and lines
   fitted with free area. For spectra of a sample, they are filled in with the settings of the
   sample's active quantification (**Use sample settings** restores them).
3. **Fit and quantify** (about 10–30 s). The plot shows the data, fit, background, background
   counts under the reference peaks and the residuals; **Zoom to line** zooms on any fitted line.
   Below are the composition (and, for spectra of a sample, the one saved in the ledger), the
   fit quality and quant flag, and the fitted peaks (energy, FWHM, area, height, P/B ratio),
   which can be downloaded as CSV.
