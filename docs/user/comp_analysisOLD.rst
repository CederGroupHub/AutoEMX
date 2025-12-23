.. _comp_analysis_tutorial:


Tutorial: EDS compositional analysis for phase identification
==============================================================

This tutorial shows how to run the automated workflow for EDS compositional analysis,
using the ``Run_Acquisition_Quant_Analysis.py`` script.

This script initiates the fully automated workflow described in Giunto `et al.`
(https://www.researchsquare.com/article/rs-7837297/v1), which includes:
    - Acquisition of EDS spectra from powder or bulk samples
    - Fit and quantification to extract compositions
    - Rule-based filtering of compositions
    - Clustering analysis to detect number of phases and extract their composition

The script allows for multiple samples to be defined and run sequentially with a *single click*.

The output will include:
    - SEM images of every analysed region,
    - A Data.csv file containing the raw spectral data and the corresponding quantified compositions
    - A Clusters.csv file containing the composition of each identified cluster, and the potential candidate phase
    compositions, along with their confidence score


Step 1 - Open script to edit
-------------------------------

Open ``autoemxsp/scripts/Run_Acquisition_Quant_Analysis.py``.

For details on all parameters, see the :ref:`API <runners_acquire_spectra>` for the ``batch_acquire_and_analyze``
function.


Step 2 - Define Samples to Analyse
-----------------------------------
Set the common parameters that define the physical sample geometry:
    - ``sample_type``: e.g. 'powder'. Supported types: powder, bulk, powder_continuous, bulk_rough
    - ``sample_halfwidth``: half-width of sample region to analyse (e.g. that of the carbon tape) in mm.
    - ``sample_substrate_type``: e.g. 'Ctape'. Supported types: Ctape, None
    - ``sample_substrate_shape``: e.g. 'circle'. Supported types: square, circle
    - ``sample_substrate_width_mm``: SEM stub diameter, in mm
    - ``working_distance: in mm``. Approximate WD at which sample is in focus. AutoEMXSp limits autofocus around this value to avoid gross mistakes in autofocus functions.

Modify ``samples`` list to set the individual sample parameters.
Each sample is defined by a dictionary with the following keys:
   - `ID`: Sample ID. All data will be saved in results_dir, in a folder named after 'ID'.
   - `els`: Elements present in the sample that will be quantified from EDS spectra.
           Do not include the elements present only in the substrate (e.g. C from carbon tape), which are defined
           later from ``els_substrate``.
           It is also advised not to include light elements that are majorly present in the substrate and appear in the spectra,
           even if actually present in the sample (e.g. do not include C from carbonates if measu.ing particles on a C tape.)
           Doing so, may disrupt completely the quantification and cause excessively large analaytical errors.
   - `pos`: position of the sample center within the microscope, in absolute microscope stage coordinates. This can typically be deduced from `Saved Positions`.
   - `cnd`: list of candidate compositions that may be present in the sample. May be modified later when re-running clustering analysis.

Note that keys can be defined in ``autoemxsp.runners.batch_acquire_and_analyze`` and used to differentiate samples.
See *Template: Customizing Parameters Per Sample* section within the ``batch_acquire_and_analyze`` script.

 
Step 3 - Define Measurement Configurations
------------------------------------------
Several measurement parameters are constant for a given microscope, and are therefore defined elsewhere when setting up ``AutoEMXSp`` in the microscope.

There are some additional parameters that the user may want to modify:
    - ``results_dir``: Absolute or relative path to project folder, where individual folders for each sample will be created. Uses default directory ``autoemxsp/Results`` if set to None. 
    - ``beam_energy``: in keV. Ensure there exists a standard reference file for this voltage. For simplicity, it is recommended to collect standards once and use always the same voltage.
    - ``is_manual_navigation``: Whether to manually navigate the microscope to the desired frame to analyse. Typically set to False, unless the user wants to analyse a very specific region of the sample.
    - ``is_auto_substrate_detection`` Whether to activate automated detection of substrate. Only sample_substrate_type = 'Ctape' is currently supported for this option, and C-tape must appear black on a brighter support stub, e.g. Al.
    - ``auto_adjust_brightness_contrast``: Whether to use automatic adjustments of brightness and contrast. Typicall set to True. If set to False, then the following must also be defined:
        - ``contrast``: Contrast value to input in the microscope. Used only if auto_adjust_brightness_contrast = False
        - ``brightness``: Contrast value to input in the microscope. Used only if auto_adjust_brightness_contrast = False
    - ``min_n_spectra``: Minimum number of spectra after which ``AutoEMXSp`` starts checking for convergence. Only useful if ``quantify_spectra = True``.
    - ``max_n_spectra``: Target number of spectra to acquire if ``quantify_spectra = False``. If ``quantify_spectra = True``, this variable indicates the maximum number of spectra collected when convergence is not achieved.
    - ``target_Xsp_counts``: Target number of counts for each spectrum
    - ``max_XSp_acquisition_time``: Maximum allowed acquisition time for each spectrum. This should be defined in function of the detectors counts/sec, in order to ensure the measurement is interrupted if a wrong region is selected by mistake (e.g. the C tape, or a void in the sample).
    **Warning** Spectra acquisitions that are interrupted due to to this parameter, are flagged and discarded. Ensure this time is high enough for you microscope system.


Step 4 - Define whether to Quantify Spectra during Acquisition
--------------------------------------------------------------
Set ``quantify_spectra`` = False | True.
Defines whether to quantify spectra during acquisition. Quantification is parallelized, but it may be slow if the microscope computer is not computationally powerful.
In this case, it is recommended to set ``quantify_spectra = False``, and follow step 8 after EDS acquisition.

When ``quantify_spectra = True``, ``AutoEMXSp`` periodically checks for convergence, and may stop the acquisition if convergence is obtained.
**Convergence criteria**:
  - If no candidate phases are present or assigned, all clusters must have a RMS point-to-centroid distance < 2.5%.
  - If candidate phases are assigned, they must be identified with a minimum confidence > 0.8 and cluster RMS point-to-centroid distance < 3%.


Step 5 - Define Other Parameters
----------------------------------------------
These parameters will not affect the spectral acquisition, and can always be changed in a second moment, but they require re-quantifying the spectra: 
    - ``interrupt_fits_bad_spectra``: Whether to interrupt the quantification of spectra expected to lead to gross quantification errors. Tested extensively. Typically set to True to speed up quantification.
    - ``min_bckgrnd_cnts``: Minimum number of counts under a reference peak necessary for a spectrum to be accepted for clustering. Keep in mind the following:
         - Spectra not satisfying this condition are flagged (quant_flag = 8) and not quantified if ``interrupt_fits_bad_spectra = True``.
         - If ``interrupt_fits_bad_spectra = False``, they are still quantified, and filtered out later in the clustering stage.
         - If too many spectra end up being flagged, consider decreasing ``min_bckgrnd_cnts`` or increasing ``target_Xsp_counts`` in your following measurements.
         - If you want to change ``min_bckgrnd_cnts`` and requantify the spectra, you may do so faster by setting ``quantify_only_unquantified_spectra = True`` when running ``autoemxsp/scripts/Run_Quantification_Analysis.py``.


    Minimum number of counts under a reference peak necesary for a spectrum to be accepted for clustering. Can be modified later. Does not influence quantification

These parameters will not affect the spectral acquisition, nor its quantification and can always be changed in a second moment, but they require re-analysing the compositions: 
    - ``max_analytical_error_percent``: Maximum analytical error to employ to filter out compositions during clustering
    - ``quant_flags_accepted``: Quantification flags accepted during clustering. See :ref:`Quant Flags Docs <config_classes>`
    - ``max_n_clusters``: Maximum number of clusters possible in the sample. Keep this number large enough to ensure the sample is well represented, but also low enough to avoid useless long computations.
    - ``show_unused_comps_clust``: Whether to show discarded compositions (i.e., black triangles) in clustering plot

        
        
Step 6 - Define Sample-Type-Specific Configurations
--------------------------------------------------------------
Depending on the selected ``sample_type``, define the following configuration classes:
    - ``powder_meas_cfg_kwargs``: if ``sample_type = 'powder'``. Defines how to detect particles and select EDS acquisition spots.
    - ``bulk_meas_cfg_kwargs``: if ``sample_type = 'powder_continuous' | 'bulk' | 'bulk rough'``. Set dimensions to define a grid of EDS acquisition spots.

For details on these parameters, see the :ref:`Configuration Classes API <config_classes>`.



Step 7 - Launch Acquisition
-------------------------------

Script must be launched at the SEM.


**Output:**

`AutoEMXSp` creates a folder named 'ID' (as defined above), acquires the spectra and saves the following files:
    - The employed `AutoEMXSp` configurations, saved in `Comp_analysis_configs.json`.
    - The `EM_metadata.msa` file, containing metadata output by the microscope manufacturer.
    - `SEM images/`` folder, containing images of every analysed region/particle, annotated with positions and ID of the acquired EDS spectra. If .tiff, contains also the annotation-free image for post-processing.
    - The `Analysed_region.png` image captured from the microscope navigation camera, annotated with the analysed region.  Only present if ``sample_type = 'powder'``.
    - A `Data.csv` file containing the following columns:
        - `Spectrum ID`: integer number, reported in the annotated SEM images.
        - `Frame ID`: Identifier of frame, to retrieve the corresponding frame image where spectrum was acquired.
        - `Particle #`: Particle identifier, to retrieve the corresponding particle image where spectrum was acquired. Only present if ``sample_type = 'powder'``.
        - (x, y): Position of spectrum in the corresponding SEM image, in relative coordinates, as defined in the electron microscope driver at ``autoemxsp/EM_driver/your_microscope_ID``.
        - `Real_time`: Acquisition time, in seconds. Time passed from beginning to the end of the acquisition.
        - `Live_time`: Effective detector acquisition time, in seconds. Real_time stripped of detector dead time.
        - `Spectrum`: Raw spectral data.
        - `Background`: Background data fitted from microscope manufacturer. Only present if ``autoemxsp.config.defaults.use_instrument_background = True``.



Step 8 - Optional - (Re)quantify spectra
------------------------------------------
This step is performed automatically if ``quantify_spectra = True`` has been selected.

Alternatively, copy the acquired data folder to another, more performant machine (possibly with a lot of CPU cores for faster parallel quantification), and run ``autoemxsp/scripts/Run_Quantification_Analysis.py``.
This script only requires a list of the samples to quantify ``samples_ID``, and the project directory ``results_dir``.

Parameters
~~~~~~~~~~

Besides the parameters defined above, there are also the following:
    - ``run_clustering_analysis``: Whether to run the clustering analysis automatically after the quantification. Typically set to True.
    - ``num_CPU_cores``: Number of CPU cores used during fitting and quantification. If set to ``None``, `AutoEMXSp` selects automatically half the available cores.
    - ``quantify_only_unquantified_spectra``: Set to True if running on Data.csv file that has already been quantified. Used to quantify only the discarded unquantified spectra, for example after having changed the value of ``min_bckgrnd_cnts``.


**Output:**

Modifies the `Data.csv` file, adding the following columns:
    - `El_at%' : Atomic fraction reported for each sample element, as defined from `els`.
    - `El_w%' : Mass fraction reported for each sample element, as defined from `els`.
    - 'An er w%' : Analytical total error, measured in mass fraction. See paper for definition.
    - r_squared : R^2 metric for goodness of fit
    - redchi_sq : Reduced chi squared metric for goodness of fit. This is used to determine if the fit is good enough.
    - Quant_flag : Flags that identify if the quantification is reliable and, if not, why it is not reliable. See :ref:`Quant Flags Docs <config_classes>` for specific meaning of these flags.
    - Comments : For good spectra, it reports the minimum counts fitted below a reference peak. For unreliable spectra, it typically communicates the reason why they are unreliable.


Step 9 - Optional - (Re)analyse spectra
----------------------------------------

This step is performed automatically if:

- ``quantify_spectra = True`` was set during acquisition.
- ``run_clustering_analysis = True`` was set during quantification.

To run or re-run the clustering analysis of the extracted compositional data, execute:

``autoemxsp/scripts/Run_Analysis.py``

This step is not computationally intensive compared to quantification and can
be run on the same machine or on a separate workstation.

**Note:** This script processes only one sample at a time, specified via
``sample_ID``.

Parameters
~~~~~~~~~~

The script accepts the same parameters described previously for acquisition
and quantification scripts. In addition, the following clustering-specific
options are available:

- ``clustering_features`` : Choose whether to use atomic fractions
  (``'at_fr'``) or mass fractions (``'w_fr'``) as features for clustering.
  Default is used if set to ``None``.
- ``k_forced`` : Force the number of clusters to a specific integer. If set
  to ``None``, the number of clusters is loaded from ``Comp_analysis_configs.json``:
  
  - If ``k`` was forced during acquisition, this value is used unless
    ``k_finding_method`` is not ``None``.
  - If ``k`` was determined automatically during acquisition (``k = None``),
    it will be re-evaluated automatically.

- ``k_finding_method`` : Method used to determine the number of clusters.
  Only applied if ``k_forced`` is ``None``. Note that if ``k`` was forced
  during acquisition, setting ``k_finding_method`` to anything other than
  ``None`` will force ``k`` to be re-evaluated.

Plotting options
~~~~~~~~~~~~~~~~

- ``ref_formulae`` : List of candidate compositions. If ``None``, the list
  is loaded from ``Comp_analysis_configs.json``.
  
  **Warning:** Providing a list will replace the loaded list unless the
  first entry is ``""`` or ``None`` (e.g., ``ref_formulae = [,"Mn2O3"]``).

- ``els_excluded_clust_plot`` : List of elements to exclude from the 3D
  clustering plot. By default, elements are used in the order defined in
  ``els``.
- ``plot_custom_plots`` : If ``True``, use the custom plot function defined
  in ``autoemxsp/_custom_plotting.py``. Useful for publication-ready plots.
- ``show_unused_compositions_cluster_plot`` : If ``True``, display discarded
  compositions as black triangles in the clustering plot. These compositions,
  although discarded due to analytical error, may visually hint at phases
  present in the sample.

Output
~~~~~~

Running the script creates an ``Analysis`` folder with the following files:

- ``Clustering_info.json`` : Contains the clustering and quantification
  configurations used.
- ``Silhouette_plot.png`` : If ``k`` was not forced, shows silhouette
  scores for the determined number of clusters.
- ``Clustering_plot.png`` : 3D clustering plot (also displayed interactively
  when the script runs).
- ``Clusters.csv`` : One row per identified cluster, with the following columns:
  
  - ``Cluster ID`` : Identifier of the cluster.
  - ``n-points`` : Number of points in the cluster.
  - ``El_at%`` : Atomic fraction of cluster centroid.
  - ``El_std_at%`` : Standard deviation of atomic fractions.
  - ``El_w%`` : Mass fraction of cluster centroid.
  - ``El_std_w%`` : Standard deviation of mass fractions.
  - ``RMS_dist_at%`` : Root-mean-square distance of points from centroid
    in atomic fraction space.
  - ``RMS_dist_w%`` : Root-mean-square distance in mass fraction space.
  - ``wcss`` : Within-cluster sum of squares (in the feature space used).
  - ``cnd`` : Identified candidate composition with confidence score
    ``CS_cnd``.
  - ``mix`` : Pair of compositions potentially intermixed, with:
    
    - ``CS_mix`` : Confidence score of mixture.
    - ``Mol_Ratio`` : Molar ratio (first phase / second phase).
    - ``X1_mean`` : Mean molar fraction of the first phase.
    - ``X1_stdev`` : Standard deviation of the first phase molar fraction.

- ``Compositions.csv`` : Similar to ``Data.csv`` but with an additional
  ``Cluster ID`` column indicating the cluster assignment.

    
    
TODO: ADD IMAGES    
CHANGE LINK TO QUANT FLAGS (2 links)
And also to k_finding_method

