<div align="center">

# AutoEMXSp 

[![PyPI version](https://badge.fury.io/py/autoemxsp.svg)](https://pypi.org/project/autoemxsp/)
[![Python Version](https://img.shields.io/pypi/pyversions/autoemxsp.svg)](https://pypi.org/project/autoemxsp/)
[![License: Custom Non-Commercial](https://img.shields.io/badge/license-Custom%20Non--Commercial-blue.svg)](https://github.com/CederGroupHub/AutoEMXSp/blob/main/LICENSE.txt)
[![Research Square Preprint](https://img.shields.io/badge/Research%20Square-preprint-orange)](https://doi.org/10.21203/rs.3.rs-7837297/v2)  
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://cedergrouphub.github.io/AutoEMXSp/)
[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://pypi.org/project/autoemxsp/)


**Automated Electron Microscopy X-Ray Spectroscopy for Compositional Characterization of Materials**

</div>

AutoEMXSp is a **fully automated framework** for SEM-EDS workflows — from spectral acquisition and quantification, to principled filtering and compositional analysis — all in **one click**.

🎥 Watch AutoEMXSp in action on a desktop SEM-EDS system at https://youtu.be/Bym58gNxlj0

🧪 Test AutoEMXSp to fit and quantify EDS spectra you have collected on your commercial SEM-EDS system (see [Tutorials](https://cedergrouphub.github.io/AutoEMXSp/user/tutorials.html)).

📖 This work is described in:  
A. Giunto *et al.*, *Accurate SEM‑EDS Quantification, Automation, and Machine Learning Enable High‑Throughput Compositional Characterization of Powders*, 2025.  
DOI: [https://doi.org/10.21203/rs.3.rs-7837297/v2](https://doi.org/10.21203/rs.3.rs-7837297/v2)

### ✨ Key Features
- **Fully automated SEM-EDS phase-level compositional analysis workflow**, which includes:
    - **Acquisition of EDS spectra**, including particle localization if sample is powder. Compatible also with bulk samples, or manual navigation. 
    - **Quantification of compositions** using the peak-to-background method
    - **Rule-based filtering** of compositions to discard poorly quantified spectra from the analysis
    - **Unsupervised machine learning–based analysis** to identify the compositions of individual phases in the sample  

- Scripts for **fitting and quantification** of single EDS spectra exported by proprietary commercial software (.msa, .emsa, .txt files)

- **Automated experimental standard collection** scripts

- **Automated particle size distribution measurements** scripts

- **Extensible architecture** — adaptable to other techniques such as  
  - Wavelength Dispersive Spectroscopy (WDS)  
  - Scanning Transmission Electron Microscopy (STEM) with EDS  

- **Extensible hardware support** — includes driver for ThermoFisher Phenom Desktop SEM series, and can be extended to any electron microscope with a Python API  

### 📊 Performance
- **Benchmarked** on 74 single-phase samples with compositions spanning **38 elements** (from nitrogen to bismuth), it achieved **<5–10% relative deviation** from expected values  
- **Machine learning** compositional analysis detects individual phase composition in **multi-phase samples**, including minor phases
- **Intermixed phases** can also be resolved

### 🧪 Supported Use Cases
- Scanning Electron Microscopy (SEM) with Energy-Dispersive Spectroscopy (EDS)  
- Powders and rough samples, e.g. rough films, or pellets, with automated segmentation.
- Bulk, flat samples, navigated by defining a grid of points to probe.
- Manual navigation of any sample.

### ⚙️ Requirements
- Cross-platform: runs on **Linux, macOS, and Windows**
- Quick installation  
- Requires calibration for use with different electron microscopes for maximum compositional accuracy, though EDS quantification can be tested without recalibration.

---

## 📑 Table of Contents
- [📘 Documentation](#-documentation)
- [📦 Requirements](#-requirements)
- [🆕 Coming Soon](#-coming-soon)
- [📂 Project Structure](#-project-structure)
- [📁 Scripts](#-scripts)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📖 Citation](#-citation)
- [📂 Paper Data](#-paper-data)
- [📬 Contact](#-contact)

---

## 📘 Documentation

Installation instructions, usage examples, and workflow descriptions are available in the AutoEMXSp documentation:

👉 https://cedergrouphub.github.io/AutoEMXSp/


---

## 📦 Requirements

- **Python 3.11 or newer**  
- All dependencies are installed automatically via `pip` or `conda`.  
- Tested versions of dependencies are specified in `pyproject.toml`.  
  > The package **may work** with more recent versions, but these have **not been tested**.
  
---

### Electron Microscope Support
- ✅ Developed and tested for **Thermo Fisher Phenom Desktop SEMs**.  
- ✅ Compatible with any Phenom microscope equipped with **PPI (Phenom Programming Interface)**.  
- ⚠️ For other microscope models, the driver must be adapted to the appropriate API commands.  


---

## 🆕 Coming Soon
Here’s what’s planned for future releases of **AutoEMXSp**:
- 🐍 Verify with the latest **Python** version for improved compatibility with current scientific libraries
- 📏 New scripts for **spectral parameter calibration** to extend the `XSp_calibs` library to your own instrument.

---

## 📂 Project Structure

The repository is organized as follows:

```text
AutoEMXSp/
├── autoemxsp/                 # Main package source code
│   ├── _custom_plotting.py     # Customizable clustering plot function
│   ├── config/                 # Configuration files, including default values to employ during measurements.
│   ├── core/                   # Core objects and source code
│   ├── data/                   # Libraries of X-ray data
│   ├── EM_driver/              # Electron Microscope driver (⚠️ adapt to your own instrument)
│   └── Results/                # Example acquired data (used for unit tests)
│   ├── runners/                # Runner functions calling on core objects
│   ├── scripts/                # Scripts to run acquisition, quantification, etc. (see full list below)
│   ├── XSp_calibs/             # X-ray spectral calibrations (⚠️ adapt to your own instrument)
│   ├── utils/                  # Utility functions and strings employed by the program
│
├── examples/                  # Example scripts for fitting, quantification and compositional analysis of example data
├── tests/                     # Unit tests for fitting, quantification, compositional analysis and image processing
│                               # (Acquisition tests require proper EM drivers)
├── paper_data/                # Raw paper data uploaded on Git LFS (Dowload instructions in Paper Data section below)
│
├── LICENSE.txt
├── README.md
└── pyproject.toml
```

---

## 📁 Scripts

This repository includes a collection of scripts that streamline the use of **AutoEMXSp**.  
Each script is tailored for a specific task in spectral acquisition, calibration, quantification, or analysis.

### 🔬 Acquisition, Quantification & Analysis
- **Run_Acquisition_Quant_Analysis.py** — Acquire X-ray spectra and optionally perform quantification and composition analysis.  
- **Run_Quantification_Analysis.py** — Quantify acquired spectra (single or multiple samples) and perform machine-learning analysis.  
- **Run_Analysis.py** — Launch customized machine-learning analysis on previously quantified data. 
- **Fit_Quant_Single_AutoEMXSp_Spectrum.py** — Fit and optionally quantify a single spectrum measured with AutoEMXSp. Prints fitting parameters and plots fitted spectrum for detailed inspection of model performance.  
- **Fit_Quant_Single_MSA_Spectrum.py** — Fit and optionally quantify a single spectrum exported by proprietary software. 

### 📊 Particle Size Distribution Measurements
- **Collect_Particle_Statistics.py** - Analyse sample, collecting particle size statistics and distribution.
- **Process_Particle_Stats_Files.py** - Process acquired aprticle size data and recompute.

### 🛠️ Miscellaneous
- **Run_Experimental_Standard_Collection.py** — Acquire and fit experimental standards.  
- **Run_SDD_Calibration.py** — Perform calibration of the SDD detector.

### ⚗️ Characterize Extent of Intermixing in Known Powder Mixtures  
*(see [Chem. Mater. 2025, 37, 6807−6822](https://pubs.acs.org/doi/10.1021/acs.chemmater.5c01573) for example)*  
Use the same scripts as regular composition characterization, as described in the docs Tutorial.

👉 All scripts can be executed directly from the command line or imported into a Python environment, making them accessible from anywhere on your system.  

---

## 🤝 Contributing

Contributions are welcome!

Open to collaborations to extend this package to different tools or to different types of samples, for example thin films.
Please contact me at agiunto@lbl.gov

---

## 📄 License

This project is licensed under a NON-COMMERCIAL USE ONLY,
LICENSE — see the LICENSE file for details.

---

## 📖 Citation

If you use **AutoEMXSp** in your research, please cite the following publication:

> A. Giunto, Y. Fei, P. Nevatia, B. Rendy, N. Szymanski and G. Ceder;
> *Accurate SEM‑EDS Quantification, Automation, and Machine Learning Enable High‑Throughput Compositional Characterization of Powders*, 2025.  
> DOI: [https://doi.org/10.21203/rs.3.rs-7837297/v1](https://doi.org/10.21203/rs.3.rs-7837297/v2)

### BibTeX
```bibtex
@article{Giunto2025AutoEMXSp,
  author  = {Giunto, Andrea and Fei, Yuxing and Nevatia, Pragnay and Rendy, Bernardus and Szymanski, Nathan and Ceder, Gerbrand},
  title   = {Accurate SEM‑EDS Quantification, Automation, and Machine Learning Enable High‑Throughput Compositional Characterization of Powders},
  year    = {2025},
  doi     = {10.21203/rs.3.rs-7837297/v2},
  url     = {https://doi.org/10.21203/rs.3.rs-7837297/v2}
}
```

---

## 📂 Paper Data

The raw data used in the associated publication is stored in the `paper_data/` directory.  
These files are tracked with **Git LFS** (Large File Storage).

### 🔽 Download with Git LFS
The repository is automatically cloned without Git LFS; you will only see placeholder files instead of the actual datasets inside `paper_data/`.  
To download the full data, on the terminal go to the repo directory and:

```bash
# 1. Install Git LFS (only needed once per machine)
git lfs install

# 2. Fetch the data files
git lfs fetch --all
git lfs checkout

```

Alternatively, download manually from the github repo Download button.

After downloading, run the Run_Analysis.py or Run_Quantification.py scripts within the folder.

---

## 📬 Contact

For questions or issues, please open an issue on GitHub.


---

