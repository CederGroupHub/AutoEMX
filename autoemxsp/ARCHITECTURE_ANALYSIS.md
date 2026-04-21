# AutoEMXSp Codebase Architecture Analysis

## Executive Summary

AutoEMXSp is a sophisticated system for automated Electron Microscopy X-ray Spectroscopy (EMXSp) analysis. The architecture centers on **orchestration classes** (`EMXSp_Composition_Analyzer`, `EM_Controller`) that coordinate specialized analysis modules (`XSp_Fitter`, `XSp_Quantifier`), with structured configuration through dataclasses. Data flows primarily as **dictionaries with string keys** rather than structured objects, which presents opportunities for Pydantic schema modernization.

---

## 1. Main Classes and Responsibilities

### **Core Analysis Classes**

#### **XSp_Fitter** (`core/XSp_fitter.py`)
- **Responsibility**: Spectral fitting and peak modeling
- **Main Methods**: 
  - `__init__()`: Initialize fitter with spectrum data and instrument parameters
  - `fit_spectrum()`: Execute the fitting algorithm; returns `(fit_result, fitted_lines)` tuple
- **Dependencies**: Background_Model, Peaks_Model, DetectorResponseFunction
- **Key Inputs**:
  - `spectrum_vals`: measured intensity array
  - `energy_vals`: energy scale
  - `beam_energy`, `emergence_angle`: experimental conditions
  - `els_to_quantify`: list of elements
  - `is_particle`: bool indicating if sample is a particle
- **Key Outputs**:
  - `fit_result`: lmfit ModelResult object containing fitted parameters
  - `fitted_lines`: dict of fitted X-ray lines with peak information

#### **XSp_Quantifier** (`core/XSp_quantifier.py`)
- **Responsibility**: Spectral quantification and composition calculation
- **Main Methods**:
  - `__init__()`: Initialize quantifier with spectrum, calibration, and sample info
  - `quantify_spectrum()`: Fit spectrum and convert peaks to composition; returns `(quant_result, min_bckgrnd_ref_lines, bad_quant_flag)` tuple
  - `initialize_and_fit_spectrum()`: Single-iteration fit (used for standards)
- **Workflow**: Creates internal `XSp_Fitter` instance → fits spectrum → extracts composition
- **Key Attributes**: 
  - Stores fitted elements, peak info, background values
  - Self-manages spectrum fitting via contained `XSp_Fitter` instance
- **Returns**:
  - `quant_result`: dict with composition data (see Data Structures section)
  - `bad_quant_flag`: int indicating quantification quality issues

#### **Quant_Corrections** (`core/XSp_quantifier.py`)
- **Responsibility**: Matrix corrections (Z, A, R factors) for quantification
- **Instance Methods**: Calculate physical correction factors
- **Used By**: `XSp_Quantifier` for converting fitted peak intensities to compositions

#### **EMXSp_Composition_Analyzer** (`core/EMXSp_composition_analyser.py`)
- **Responsibility**: Master orchestrator for complete analysis workflow
- **Main Methods**:
  - `__init__()`: Initialize with all configuration objects
  - `run_collection_and_quantification()`: Main workflow entry point
  - `_quantify_all_spectra()`: Parallel quantification of collected spectra
  - Clustering, phase identification, plotting, and results export
- **Key Workflow**:
  1. Initialize microscope/detector via `EM_Controller`
  2. Collect spectra via `EM_Particle_Finder` 
  3. Quantify all spectra in parallel using `XSp_Quantifier`
  4. Cluster compositions, identify phases, export results
- **Stores**: `spectral_data`, `spectra_quant`, `sp_coords` (accumulated from all collections)

### **Automation Classes**

#### **EM_Controller** (`core/EM_controller.py`)
- **Responsibility**: Microscope hardware control and X-ray spectra acquisition
- **Main Methods**:
  - `initialise_SEM()`: Wake up microscope, set beam parameters
  - `initialise_XS_analyzer()`: Get EDS/WDS analyzer object
  - `acquire_XS_spot_spectrum(x, y)`: Collect spectrum at position
  - Stage movement, focus, brightness/contrast adjustment
- **Configuration**: Takes `MicroscopeConfig`, `MeasurementConfig`, `SampleConfig` objects
- **Interactions**: 
  - Creates/uses `EM_Particle_Finder` for automated particle analysis
  - Returns raw spectral data (counts array, collection time)

#### **EM_Particle_Finder** (`core/EM_particle_finder.py`)
- **Responsibility**: Particle detection, selection, and X-ray spot planning
- **Main Methods**:
  - `go_to_next_particle()`: Navigate to next particle
  - `get_XS_acquisition_spots_coord_list()`: Determine spot coordinates on current particle
  - `get_particle_stats()`: Collect particle size statistics
- **Configuration**: `PowderMeasurementConfig` with particle detection/filtering params
- **Outputs**: Particle positions, acquisition spot coordinates

#### **EM_Sample_Finder** (`core/EM_controller.py`)
- **Responsibility**: Sample location detection (e.g., carbon tape center detection)

---

## 2. Configuration Architecture (Config Classes)

All configuration uses **frozen dataclasses** in `config/classes.py`:

| Class | Purpose | Key Fields |
|-------|---------|-----------|
| **MicroscopeConfig** | Hardware/detector setup | `ID`, `type` (SEM/STEM), `is_auto_BC`, `energy_zero`, `bin_width` |
| **SampleConfig** | Sample identity & spatial | `ID`, `elements`, `type` (powder/bulk/film), `center_pos`, `half_width_mm` |
| **SampleSubstrateConfig** | Substrate composition | `elements`, `type` (Ctape), `shape` (circle/square) |
| **MeasurementConfig** | Acquisition settings | `type` (EDS/WDS), `mode`, `beam_energy_keV`, `emergence_angle`, acquisition time/count targets |
| **QuantConfig** | Fitting & quantification | `method` (PB), `spectrum_lims`, `fit_tolerance`, `use_instrument_background` |
| **PowderMeasurementConfig** | Powder-specific settings | `par_search_frame_width_um`, `max_n_par_per_frame`, particle segmentation model |
| **BulkMeasurementConfig** | Bulk-specific settings | Grid spacing and sampling parameters |
| **ClusteringConfig** | Clustering & phase ID | `ref_formulae`, clustering method, confidence thresholds |
| **PlotConfig** | Visualization | Save paths, plot formats, display options |

**Key Pattern**: Config objects are passed to all major classes via `__init__()`, providing unified parameter management.

---

## 3. Main Data Structures and Data Flow

### **3.1 Quantification Result Dictionary** (`quant_result`)

**Structure** (returned by `XSp_Quantifier.quantify_spectrum()`):
```python
{
    cnst.COMP_AT_FR_KEY: {           # 'compositions_at_fr'
        'Fe': 0.35,
        'O': 0.65,
        ...
    },
    cnst.COMP_W_FR_KEY: {            # 'compositions_w_fr'
        'Fe': 0.49,
        'O': 0.51,
        ...
    },
    cnst.AN_ER_KEY: 0.05,            # 'analytical_error' (proportion)
    cnst.REDCHI_SQ_KEY: 2.3,         # 'redchi_sq'
    cnst.R_SQ_KEY: 0.98              # 'r_sq'
}
```

**Passed Between**:
- `XSp_Quantifier.quantify_spectrum()` → `EMXSp_Composition_Analyzer._quantify_single_spectrum()`
- Accumulated in `EMXSp_Composition_Analyzer.spectra_quant` list
- Used for clustering, phase identification, plotting

### **3.2 Spectral Data Accumulator** (`spectral_data`)

**Structure** (in `EMXSp_Composition_Analyzer`):
```python
self.spectral_data = {
    cnst.SPECTRUM_DF_KEY: [counts_array_1, counts_array_2, ...],          # Spectrum intensity
    cnst.BACKGROUND_DF_KEY: [bg_array_1, bg_array_2, ...],                # Background
    cnst.REAL_TIME_DF_KEY: [10.5, 11.2, ...],                             # Real time (s)
    cnst.LIVE_TIME_DF_KEY: [9.8, 10.1, ...],                              # Live time (s)
    cnst.COMMENTS_DF_KEY: ["spectrum ok", "low counts", ...],             # QC comments
    cnst.QUANT_FLAG_DF_KEY: [None, 1, ...]                                # Error flags
}
```

**Data Flow**:
1. Each acquired spectrum appended via `_store_raw_spectral_data()` 
2. During quantification, background values updated via `_assemble_fit_info()`
3. Comments and flags added after each quantification attempt
4. Saved to CSV at end of workflow

### **3.3 Fitted Spectrum Components Dictionary**

**From `fit_result.eval_components(x=energy_vals)`**:
```python
fit_components = {
    'background': background_counts_array,
    'Fe_Ka1': peak_counts_array,
    'Fe_Kb1': peak_counts_array,
    'O_Ka1': peak_counts_array,
    ...
}
```

**Source**: `XSp_Fitter.fit_spectrum()` → stored in `XSp_Quantifier.fit_components`

### **3.4 Fit Results Dictionary** (Peak/Background Info)

**Structure** (returned by `_assemble_fit_info()`):
```python
fit_results_dict = {
    'Fe_Ka1_PB_ratio': 0.75,          # Peak-to-background ratio
    'O_Ka1_PB_ratio': 0.32,
    ...,
    cnst.R_SQ_KEY: 0.97,              # Goodness-of-fit metrics
    cnst.REDCHI_SQ_KEY: 1.8
}
```

**Usage**: Metadata stored with quantification results for quality assessment

### **3.5 Spectrum Data Dictionary** (Acquired at Microscope)

**From `EM_Controller.acquire_XS_spot_spectrum(x, y)`** (raw data):
```python
spectral_data_raw = {
    'spectrum': np.array([counts]),         # Intensity array
    'real_time': float,                     # Acquisition time
    'live_time': float,                     # Detector live time
    # Optionally:
    'background': np.array([bg_counts]),    # Instrument background (if available)
}
```

---

## 4. Main Workflows

### **Workflow A: Full Collection and Quantification** 
(Entry: `EMXSp_Composition_Analyzer.run_collection_and_quantification()`)

```
Initialize Configurations
        ↓
[EMXSp_Composition_Analyzer.__init__()]
    ├→ EM_Controller.initialise_SEM()
    ├→ EM_Particle_Finder.__init__()
    └→ Load microscope calibrations
        ↓
For Each Particle (automated or manual):
    ├→ EM_Particle_Finder.go_to_next_particle()
    ├→ EM_Particle_Finder.get_XS_acquisition_spots_coord_list()
    │
    └─→ For Each Spot:
        ├→ EM_Controller.acquire_XS_spot_spectrum(x, y)
        │   Returns: {spectrum, real_time, live_time, background}
        │
        └→ EMXSp_Composition_Analyzer._store_raw_spectral_data()
            └→ Append to self.spectral_data dict
        ↓
[Quantification Phase] EMXSp_Composition_Analyzer._quantify_all_spectra()
    └→ Parallel (joblib) over all spectra:
        ├→ XSp_Quantifier.__init__()
        │   Takes: spectrum_vals, energy_vals, calibration info
        │
        ├→ XSp_Quantifier.quantify_spectrum()
        │   └→ XSp_Fitter.__init__() [internal]
        │   └→ XSp_Fitter.fit_spectrum()
        │       └→ Background_Model + Peaks_Model + lmfit
        │       Returns: (fit_result, fitted_lines)
        │   └→ Quant_Corrections: calculate composition from peak intensities
        │   Returns: (quant_result, min_bckgrnd_ref_lines, bad_quant_flag)
        │
        └→ Store: self.spectra_quant.append(quant_result)
        ↓
[Post-Processing]
    ├→ Clustering: KMeans/DBSCAN on compositions
    ├→ Phase identification: Match to reference phases
    ├→ Results export: CSV, Excel, plots
    └→ Save spectral_data to HDF5/CSV
```

### **Workflow B: Standard Quantification** 
(Entry: `XSp_Quantifier.initialize_and_fit_spectrum()`)

```
XSp_Quantifier with constrained composition (standard)
    ├→ Get initial K value (background scaling)
    ├→ XSp_Fitter.__init__()
    ├→ XSp_Fitter.fit_spectrum()
    └→ Extract fitted parameters & check reliability
```

### **Workflow C: Particle Statistics Collection**
(Entry: `EM_Particle_Finder.get_particle_stats()`)

```
EM_Particle_Finder.get_particle_stats(n_par_target=500)
    └→ Loop through frames:
        ├→ EM_Controller image acquisition
        ├→ Particle detection/segmentation
        ├→ Filter by size criteria
        ├→ Accumulate particle area statistics
        └→ Export size distribution
```

---

## 5. Parameter/Configuration Passing Patterns

### **Pattern 1: Config Object Injection**
```python
# EMXSp_Composition_Analyzer accepts all config objects
analyzer = EMXSp_Composition_Analyzer(
    microscope_cfg=MicroscopeConfig(...),
    sample_cfg=SampleConfig(...),
    measurement_cfg=MeasurementConfig(...),
    ...
)
```

### **Pattern 2: Dictionary Parameter Spreading**
```python
# Microscope calibrations loaded globally
calibs.load_microscope_calibrations(microscope_cfg.ID, measurement_cfg.mode)
# Then accessed via module attributes: calibs.emergence_angle, calibs.undetectable_els, etc.
```

### **Pattern 3: Instance Attributes for State**
```python
# XSp_Quantifier stores many parameters as instance attributes
self.beam_energy = beam_e
self.emergence_angle = emergence_angle
self.det_ch_offset = det_ch_offset
self.det_ch_width = det_ch_width
self.microscope_ID = microscope_ID
# Later used in methods without re-passing
```

### **Pattern 4: Optional Parameters with Defaults**
```python
# Many classes use None defaults with fallback logic
els_substrate = els_substrate or ['C', 'O', 'Al']
els_w_fr = els_w_fr or {}
```

### **Pattern 5: Kwargs Pattern** (Minimal Use)
- Generally avoided; explicit parameters preferred
- Some use in utility functions for flexibility

---

## 6. Data Type Catalog

### **Arrays & Series**
| Type | Context | Example |
|------|---------|---------|
| `np.ndarray` | Spectrum intensity | `spectrum_vals[i]` |
| `np.ndarray` | Energy scale | `energy_vals[i]` (1D) |
| `np.ndarray` | Fit components | `fit_components['Fe_Ka1']` |
| `pd.DataFrame` | Quantification results | Exported to CSV |
| `pd.Series` | Composition data | `quant_result[cnst.COMP_W_FR_KEY]` |

### **Dictionaries (Critical for Refactoring)**
| Key Pattern | Value Type | Purpose |
|-------------|-----------|---------|
| `cnst.COMP_AT_FR_KEY` | `Dict[str, float]` | Atomic fractions by element |
| `cnst.COMP_W_FR_KEY` | `Dict[str, float]` | Weight fractions by element |
| Spectra-level | `Dict[str, Any]` | Peak info, metadata |
| Calibration | `Dict` | Microscope/detector params |

### **lmfit Objects**
| Class | Purpose |
|-------|---------|
| `lmfit.Parameters` | Fit parameter definitions |
| `lmfit.ModelResult` | Fit output with parameters, residuals, covariance |
| `lmfit.Model` | Composite model (background + peaks) |

### **Configuration**
| Type | Purpose |
|------|---------|
| `dataclass` | Structured config (MicroscopeConfig, etc.) |
| `Dict[str, float]` | Standards reference values |
| `Dict[str, str]` | Element symbol lookups |

---

## 7. Areas Where Pydantic Schemas Would Be Beneficial

### **High Priority** ✓

1. **Quantification Result** (`quant_result`)
   ```python
   # Current: Dict with string keys (error-prone)
   quant_result = {
       'compositions_at_fr': {...},
       'compositions_w_fr': {...},
       'analytical_error': 0.05,
       ...
   }
   
   # Proposed Pydantic:
   from pydantic import BaseModel, Field
   
   class QuantificationResult(BaseModel):
       compositions_at_fr: Dict[str, float] = Field(..., description="Atomic fractions by element")
       compositions_w_fr: Dict[str, float] = Field(..., description="Weight fractions by element")
       analytical_error: float = Field(..., ge=0, le=1)
       redchi_sq: float = Field(..., gt=0)
       r_sq: float = Field(..., ge=0, le=1)
   ```
   **Benefits**: 
   - Type validation at creation
   - IDE autocomplete for results
   - Clear field documentation
   - Automatic JSON serialization for export

2. **Spectral Data Accumulator** (`spectral_data`)
   ```python
   class SpectralDataAccumulator(BaseModel):
       spectrum: List[np.ndarray]
       background: List[Optional[np.ndarray]]
       real_time: List[float]
       live_time: List[float]
       comments: List[str]
       quant_flag: List[Optional[int]]
       
       class Config:
           arbitrary_types_allowed = True  # For numpy arrays
   ```
   **Benefits**:
   - Ensures all lists stay synchronized
   - Validates list lengths match
   - Clear data format documentation

3. **Fit Results Dictionary** (`fit_results_dict`)
   ```python
   class FitResultsInfo(BaseModel):
       peak_to_background_ratios: Dict[str, float]
       r_squared: float
       reduced_chi_square: float
   ```

4. **Raw Spectral Data** (from microscope)
   ```python
   class RawSpectralData(BaseModel):
       spectrum: np.ndarray = Field(..., description="Intensity counts array")
       background: Optional[np.ndarray] = None
       real_time: float = Field(..., gt=0)
       live_time: float = Field(..., gt=0)
       x_mm: float = Field(..., description="Stage X position")
       y_mm: float = Field(..., description="Stage Y position")
   ```

### **Medium Priority** ✓

5. **Fitted Peak Information**
   - Current: Scattered across lmfit Parameters and custom dicts
   - Schema: `FittedPeak` with energy, intensity, FWHM, background

6. **Composition Analysis Result** (for clustering/phase ID)
   ```python
   class CompositionAnalysisResult(BaseModel):
       composition: QuantificationResult
       particle_id: int
       spot_id: int
       phase_identified: Optional[str]
       confidence: float
       timestamp: datetime
   ```

7. **Particle Detection Result**
   ```python
   class ParticleDetectionResult(BaseModel):
       particle_id: int
       centroid_x_um: float
       centroid_y_um: float
       area_um2: float
       frame_id: int
       acquisition_spots: List[Tuple[float, float]]
   ```

### **Lower Priority** 
- Standards dictionary (already dict[str, float])
- Clustering results (use sklearn outputs directly)
- Plotting parameters (lightweight config)

---

## 8. Class Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│          EMXSp_Composition_Analyzer (Orchestrator)               │
│  - Workflow management                                            │
│  - Result aggregation & export                                   │
│  - Clustering & phase identification                             │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ├──→ EM_Controller (Hardware Control)
           │    ├─→ EM_Particle_Finder (Particle Detection)
           │    │    └─→ Particle segmentation models
           │    │
           │    └─→ EM_driver (Low-level microscope control)
           │
           ├──→ XSp_Quantifier (Quantification Orchestrator)
           │    ├─→ XSp_Fitter (Spectral Fitting)
           │    │    ├─→ Background_Model
           │    │    ├─→ Peaks_Model
           │    │    └─→ DetectorResponseFunction
           │    │
           │    └─→ Quant_Corrections (Physical calculations)
           │
           ├──→ Configuration Objects (Inputs)
           │    ├─→ MicroscopeConfig
           │    ├─→ SampleConfig
           │    ├─→ MeasurementConfig
           │    ├─→ QuantConfig
           │    ├─→ PowderMeasurementConfig
           │    ├─→ ClusteringConfig
           │    └─→ PlotConfig
           │
           └──→ Calibration Module (calibs)
                ├─→ Microscope-specific calibrations
                ├─→ Detector response functions
                └─→ Undetectable elements list
```

---

## 9. Data Flow at Different Scales

### **Micro (Single Spectrum)**
```
Raw Spectrum Counts
    ↓ (detector energy calibration)
Energy-calibrated Spectrum
    ↓ (XSp_Fitter)
Fitted Peak Intensities + Background
    ↓ (XSp_Quantifier.Quant_Corrections)
Composition Dict (at% & wt%)
    ↓ (Store in quant_result)
Quantification Result
```

### **Meso (Single Particle)**
```
Multiple Spots on Particle
    ↓ (XSp_Quantifier × N)
Multiple Quantification Results
    ↓ (Average or filter)
Particle Average Composition
    ↓ (Store with metadata)
```

### **Macro (Full Sample)**
```
All Particles in Sample
    ↓ (Parallel quantification)
All Quantification Results
    ↓ (accumulate in spectra_quant)
Composition Distribution
    ↓ (Clustering + Phase ID)
Phase Map + Statistics
    ↓ (Export)
Final Results (CSV, plots, etc.)
```

---

## 10. Key Insights for Architecture Improvement

### **Current Strengths**
1. ✓ Clear separation of concerns (fitting vs quantification vs orchestration)
2. ✓ Modular design allows swapping components (e.g., segmentation models)
3. ✓ Structured configuration via dataclasses
4. ✓ Excellent docstring documentation

### **Improvement Opportunities**
1. **Replace Dict-based results with Pydantic models** → Type safety, validation, IDE support
2. **Extract magic string keys** → Already partially done with `constants` module
3. **Reduce state in instance attributes** → Consider passing context objects instead
4. **Parallel quantification** → Already using joblib, but could benefit from result type validation
5. **Error handling** → Standardize exception types (currently mixed custom + built-in)
6. **Testing** → Data validation in Pydantic would reduce test burden

### **Recommended Refactoring Priority**
1. `QuantificationResult` schema (used everywhere)
2. `SpectralData` accumulator (prevents silent bugs)
3. `RawSpectralData` schema (interface to microscope)
4. Migrate remaining dict returns to dataclasses/Pydantic

---

## 11. Constants and Naming Conventions

**Accessed via `cnst` module**:
- `COMP_AT_FR_KEY` = 'compositions_at_fr'
- `COMP_W_FR_KEY` = 'compositions_w_fr'
- `AN_ER_KEY` = 'analytical_error'
- `REDCHI_SQ_KEY` = 'redchi_sq'
- `R_SQ_KEY` = 'r_sq'
- `SPECTRUM_DF_KEY`, `BACKGROUND_DF_KEY`, `LIVE_TIME_DF_KEY`, etc.

**Undetectable elements** (filtered automatically):
- Stored in `calibs.undetectable_els`
- Removed from quantification lists before processing

---

## Summary Table: Class Responsibilities

| Class | Tier | Key Responsibility | Returns |
|-------|------|-------------------|---------|
| **EMXSp_Composition_Analyzer** | Meta | Orchestration, workflow, export | Results dict/CSV |
| **EM_Controller** | Hardware | Microscope control, acquisition | Raw spectrum data |
| **EM_Particle_Finder** | Hardware | Particle detection, navigation | Particle positions |
| **XSp_Quantifier** | Analysis | Quantification orchestration | `quant_result` dict |
| **XSp_Fitter** | Analysis | Spectrum fitting | `fit_result` (lmfit) |
| **Background_Model** | Physics | Background continuum | Fitted background array |
| **Peaks_Model** | Physics | Peak parameterization | Peak model contribution |
| **Quant_Corrections** | Physics | Matrix corrections | Composition dict |
| **DetectorResponseFunction** | Physics | Detector modeling | Convolution matrices |
| **Config Classes** | Input | Parameter validation | None (validation only) |

