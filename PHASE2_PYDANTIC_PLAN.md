# Phase 2: Pydantic Schemas & Data Structure Refactoring

## Overview
Replace dictionary-based data flow with Pydantic models for type safety, validation, and better IDE support.

## Priority 1 Schemas (Week 1-2)

### 1. `QuantificationResult` ⭐ HIGHEST PRIORITY
**Purpose:** Output of quantitative analysis  
**Location:** To be placed in `autoemx/core/schemas.py`

```python
class QuantificationResult(BaseModel):
    element: str
    line: str  # e.g., 'Ka1', 'La1'
    area: float  # peak area from fit
    area_uncertainty: Optional[float]
    weight_fraction: float  # normalized to sum=1
    weight_fraction_uncertainty: Optional[float]
    mass_fraction: float
    atomic_fraction: float
    counts: int
    
    model_config = ConfigDict(frozen=False)
```

**Usage Impact:**
- Replaces dict returns from `XSp_Quantifier`
- Enables proper typing in return signatures
- Allows validation of weight fractions sum to 1

---

### 2. `RawSpectralData`
**Purpose:** Microscope acquisition data  
**Location:** `autoemx/core/schemas.py`

```python
class RawSpectralData(BaseModel):
    spectrum_vals: np.ndarray  # counts at each channel
    energy_vals: np.ndarray  # keV
    total_counts: int
    collection_time: float  # seconds
    beam_energy: float  # keV
    emergence_angle: float  # degrees
    microscope_id: str
    meas_mode: str  # 'point' or 'map'
    
    class Config:
        arbitrary_types_allowed = True
```

**Usage Impact:**
- Replaces scattered parameters in `XSp_Fitter.__init__`
- Single object passed to fitter
- Validates energy axis is properly sorted

---

### 3. `SpectrumFitResult`
**Purpose:** Complete fit information  
**Location:** `autoemx/core/schemas.py`

```python
class SpectrumFitResult(BaseModel):
    fit_quality: FitQuality
    parameters: Dict[str, FitParameter]
    background_model: str  # 'DuncumbMod', 'Philibert', etc
    peaks_fitted: List[str]  # ['Fe_Ka1', 'Ni_Ka1', ...]
    reduced_chi_square: float
    r_squared: float
    iterations: int
    
    @field_validator('reduced_chi_square')
    @classmethod
    def validate_chi_square(cls, v):
        if v < 0:
            raise ValueError('Chi-square must be non-negative')
        return v
```

**FitParameter Sub-Schema:**
```python
class FitParameter(BaseModel):
    value: float
    stderr: Optional[float]
    vary: bool
    bounds: Tuple[Optional[float], Optional[float]]
    expression: Optional[str]
```

**Usage Impact:**
- Replaces `lmfit.ModelResult` wrapping
- Type-safe access to fit results
- Validation of physics constraints

---

### 4. `SpectralDataBatch`
**Purpose:** Collection of spectra for batch processing  
**Location:** `autoemx/core/schemas.py`

```python
class SpectralDataBatch(BaseModel):
    spectra: List[RawSpectralData]
    sample_name: str
    acquisition_date: datetime
    operator: Optional[str]
    notes: Optional[str]
    
    @field_validator('spectra')
    @classmethod
    def validate_consistent_axes(cls, v):
        if len(v) > 1:
            first_energy = v[0].energy_vals
            for spectrum in v[1:]:
                if not np.allclose(spectrum.energy_vals, first_energy):
                    raise ValueError('All spectra must have identical energy axes')
        return v
```

**Usage Impact:**
- Enables batch quantification with validation
- Single entry point for `XSp_Quantifier.quantify_batch()`

---

## Priority 2 Schemas (Week 2-3)

### 5. `CalibrationConfig`
```python
class CalibrationConfig(BaseModel):
    microscope_id: str
    meas_mode: str
    detector_name: str
    window_type: str  # 'AP3.2', etc
    peak_shape_params: Dict[str, PeakShapeParams]
    background_params: Dict[str, float]
    efficiency_curve: np.ndarray
```

### 6. `ElementQuantification`
```python
class ElementQuantification(BaseModel):
    element: str
    primary_line: QuantificationResult
    secondary_lines: List[QuantificationResult]
    total_intensity: float
    detection_limit: float
    
    @property
    def best_result(self) -> QuantificationResult:
        """Return primary line result"""
        return self.primary_line
```

### 7. `SampleComposition`
```python
class SampleComposition(BaseModel):
    elements: Dict[str, ElementQuantification]
    total_weight_fraction: float
    
    @field_validator('total_weight_fraction')
    @classmethod
    def validate_normalization(cls, v):
        if not 0.98 < v <= 1.01:  # Allow 2% error
            raise ValueError(f'Total weight fraction {v} not normalized to 1')
        return v
```

---

## Implementation Strategy

### Phase 2a: Schema Definitions (Days 1-3)
1. Create `autoemx/core/schemas.py`
2. Define Priority 1 schemas with full validation
3. Add custom validators for physics constraints
4. Create utility functions for schema conversion

### Phase 2b: Integration Layer (Days 4-7)
1. Create conversion functions:
   ```python
   def dict_to_quantification_result(fit_dict) -> QuantificationResult
   def lmfit_result_to_spectrum_fit(lmfit_result) -> SpectrumFitResult
   ```
2. Update `XSp_Quantifier.quantify_spectrum()` return type
3. Update `XSp_Fitter` to accept `RawSpectralData` parameter
4. Add optional validation flag (for performance in batch mode)

### Phase 2c: Dataclass Refactoring (Week 3)
1. Convert `Peaks_Model` to `@dataclass` with methods
2. Convert `Background_Model` to `@dataclass`
3. Keep `XSp_Fitter` as regular class (orchestrator role)
4. Document memory/performance improvements

---

## Validation Rules

### QuantificationResult
- `area >= 0`
- `weight_fraction >= 0 and <= 1`
- `uncertainty >= 0` (if provided)
- Element symbol is valid (lookup against Element)

### RawSpectralData
- Energy axis strictly increasing
- Spectrum values >= 0
- Collection time > 0
- Beam energy > 0
- Emergence angle between 10-90°
- Array shapes match: `len(spectrum_vals) == len(energy_vals)`

### SpectrumFitResult
- Chi-square >= 0
- R-squared in [-∞, 1]
- Iterations > 0
- peaks_fitted is non-empty

### SpectralDataBatch
- Non-empty spectrum list
- All spectra have identical energy axes
- All spectra in same meas_mode

---

## Backward Compatibility

**Breaking Changes:**
```python
# OLD (still in XSp_quantifier.py for now)
result_dict = quantifier.quantify_spectrum(spectrum, elements=['Fe', 'Ni'])
weight_fraction = result_dict['Fe']['weight_fraction']

# NEW 
result = quantifier.quantify_spectrum(spectrum, elements=['Fe', 'Ni'])
weight_fraction = result['Fe'].weight_fraction  # Type-safe!
# OR
weight_fraction = result.get_element('Fe').weight_fraction
```

**Deprecation Plan:**
- Phase 2: Add new schema-based methods alongside old dict-based methods
- Phase 3: Mark old methods as `@deprecated`
- Phase 4: Remove dict-based methods

---

## Testing Strategy

### Unit Tests
```python
def test_quantification_result_validation():
    # Valid
    QuantificationResult(element='Fe', area=100, weight_fraction=0.5)
    
    # Invalid - raises ValidationError
    with pytest.raises(ValidationError):
        QuantificationResult(element='Fe', area=-10)
```

### Integration Tests
```python
def test_schema_roundtrip(lmfit_result, spectrum_data):
    # lmfit -> SpectrumFitResult -> dict -> SpectrumFitResult
    schema_result = SpectrumFitResult.from_lmfit(lmfit_result)
    dict_result = schema_result.model_dump()
    recovered = SpectrumFitResult(**dict_result)
    assert recovered == schema_result
```

---

## Performance Considerations

### Memory Usage
- Dataclasses: ~20% less overhead than full classes
- Pydantic v2: Optimized for validation speed
- Recommendation: Lazy validation in batch mode

### Speed Optimization
```python
# For batch processing, skip validation
class Config:
    validate_assignment = False  # During fitting
    # Re-validate after fit complete
```

---

## Success Criteria

✅ All Priority 1 schemas defined and tested  
✅ `XSp_Quantifier` updated to return `Dict[str, ElementQuantification]`  
✅ Type hints throughout codebase  
✅ 0 dict-based interface in new code  
✅ All existing tests pass (with schema wrappers)  
✅ Documentation updated with schema examples  

---

## Related Files to Update

- `autoemx/core/XSp_quantifier.py` — Return types
- `autoemx/core/fitting/fitter.py` — Accept RawSpectralData
- `autoemx/core/fitting/__init__.py` — Export schemas
- `autoemx/runners/*.py` — Pass RawSpectralData objects
- Test suite — Add schema validation tests

---

## Notes for Next Phase

1. **Pydantic v2 syntax** — Ensure all validators use new @field_validator pattern
2. **NumPy arrays in Pydantic** — Use `arbitrary_types_allowed = True` or serialize to lists
3. **Circular imports** — Keep schemas in separate module, import only what's needed
4. **Documentation** — Add schema diagrams to user guide
