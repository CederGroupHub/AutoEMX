.. AutoEMXSp documentation master file, created by
   sphinx-quickstart on Mon Dec 15 16:11:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


AutoEMXSp Documentation
==========================

Welcome to ``AutoEMXSp``, a Python package for **Automated Electron Microscopy
X-ray Spectroscopy**:

https://github.com/CederGroupHub/AutoEMXSp

`AutoEMXSp` is a framework for running **automated acquisition and analysis
routines** on electron microscopes (EM), offering both X-ray spectroscopy
and image acquisition and analysis workflows.

`AutoEMXSp` currently supports **Energy-Dispersive X-ray Spectroscopy (EDS)**
in **Scanning Electron Microscopy (SEM)**.

The package was primarily conceived for **automated EDS compositional analysis**,
but it also includes scripts for **automated particle size distribution
measurements** based on SEM imaging.

This work is described in:
    A. Giunto *et al.*, *Accurate SEM‑EDS Quantification, Automation, and
    Machine Learning Enable High‑Throughput Compositional Characterization
    of Powders*, 2025.  
    DOI: https://doi.org/10.21203/rs.3.rs-7837297/v2
    
Please cite this work if you use ``AutoEMXSp``.


Key Features
------------

- **Fully automated SEM-EDS compositional analysis of samples**, integrating:
    - **Live SEM control** for particle identification and EDS spectral acquisition
    - **EDS spectra quantification** using the peak-to-background method
    - **Rule-based filtering** of compositions to discard poorly quantified spectra
    - **Unsupervised machine learning–based compositional analysis** to identify the
      compositions of individual phases in the sample
      
- Manual single/multiple EDS spectra quantification

- **Automated experimental standard collection** scripts

- **Automated particle size distribution measurement** scripts

- **Extensible architecture**, adaptable to other techniques such as:
  - Wavelength Dispersive Spectroscopy (WDS)
  - Scanning Transmission Electron Microscopy (STEM) with EDS

- **Extensible hardware support**, including a driver for the
  ThermoFisher Phenom Desktop SEM series, and adaptable to any electron
  microscope exposing a Python API
  

Supported Use Cases
-------------------

- Powders and rough samples (e.g. rough films or pellets)
- Scanning Electron Microscopy (SEM) with Energy-Dispersive Spectroscopy (EDS)


Demo
---------------------------

Watch ``AutoEMXSp`` in action on a desktop SEM–EDS system:
https://youtu.be/Bym58gNxlj0


Performance
-----------

- **Benchmarked** on 74 single-phase samples spanning **38 elements**
  (from nitrogen to bismuth), achieving **<5–10% relative deviation** from
  expected values
  
- **Machine learning–based compositional analysis** detects individual phase
  compositions in **multi-phase samples**, including minor phases

- **Intermixed phases** can also be resolved

See https://doi.org/10.21203/rs.3.rs-7837297/v2 for more details


Requirements
------------

- Python 3.11 or above

- Electron Microscope provided with an API.
  ``AutoEMXSp`` comes with a driver for Thermofisher PyPhenom. For different microscopes, 
  the EM_driver must be adapted (see :ref:`EM Driver Set Up <advanced_new_EM_driver>`).
  
- ``AutoEMXSp`` comes with EDS calibrations for the Thermofisher Phenom XL series. Different microscopes or detectors
  require recalibration (see :ref:`Calibrating EDS <advanced_new_XSp_calibs>`).



Scope of the Documentation
--------------------------

This documentation is intended for **both standard and advanced users** of the
AutoEMXSp package.

- **Standard users**

  You run predefined scripts and calibrate the Silicon Drift Detector (SDD)
  without any prior knowledge of the internal code structure.  
  The documentation provides **step-by-step instructions** to help you get
  started quickly.


- **Advanced users**

  You interact with AutoEMXSp beyond simple script execution.  
  This documentation guides you through the **initial configuration and setup**
  required to deploy AutoEMXSp on a **new microscope** and adapt it to new
  experimental workflows.

.. toctree::
   :maxdepth: 2
   :caption: User Documentation:

   user/index
   
   
.. toctree::
   :maxdepth: 2
   :caption: Advanced User Documentation:

   advanced_user/index