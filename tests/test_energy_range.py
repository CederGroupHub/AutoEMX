"""Fitted energy range (keV): conversion to detector channels, compatibility with older ledgers, and the
numerical stability of the skewed peak tail over wide ranges."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc

from autoemx.config.schema_models.quantification import QuantificationConfig, _canonicalize_json_value
from autoemx.core.fitter.peaks import Peaks_Model
from autoemx.utils.helper import energy_range_to_channel_lims, resolve_energy_range_keV

INPUTS = Path(__file__).parent / "inputs"
OLD_LEDGERS = ["Wulfenite_mini", "SDD_calibration_Al", "K412_cluster_mini"]  # quantifications stored as channels

# PhenomXL point-mode calibration (keV) from which the default overvoltage was derived
PHENOM_OFFSET, PHENOM_WIDTH = -0.032459, 0.009934


# ----------------------------------------------------------------------------- energy range -> channels
def test_default_range_reproduces_former_phenom_channels_at_15_kV():
    assert energy_range_to_channel_lims(None, PHENOM_OFFSET, PHENOM_WIDTH, 15.0) == (14, 1100)


def test_upper_energy_scales_with_beam_energy():
    low, high = resolve_energy_range_keV((0.1, None), 20.0)
    assert low == 0.1
    assert high == pytest.approx(20.0 / 1.377)


def test_channels_never_reach_zero_energy_with_negative_offset():
    # Oxford-like calibration: channel 20 is at 0 keV
    sp_start, sp_end = energy_range_to_channel_lims((0.1, None), -0.2, 0.01, 20.0, n_channels=2048)
    assert (sp_start, sp_end) == (30, 1473)
    assert -0.2 + 0.01 * sp_start == pytest.approx(0.1)


def test_channels_clipped_to_spectrum_length():
    assert energy_range_to_channel_lims((0.1, 30.0), -0.2, 0.01, 30.0, n_channels=2048)[1] == 2048


@pytest.mark.parametrize("energy_range", [(0.0, 10.0), (5.0, 2.0), (-1.0, None)])
def test_invalid_ranges_rejected(energy_range):
    with pytest.raises(ValueError):
        energy_range_to_channel_lims(energy_range, PHENOM_OFFSET, PHENOM_WIDTH, 15.0)


# ----------------------------------------------------------------------------- older ledgers
def _stored_quant_config(sample: str) -> dict:
    ledger = json.loads((INPUTS / sample / "ledger.json").read_text())
    entry = ledger["quantifications"][0]
    return entry.get("config", entry)


@pytest.mark.parametrize("sample", OLD_LEDGERS)
def test_fingerprint_of_channel_limits_unchanged(sample):
    """Quantifications stored with channel limits keep the fingerprint of the former formula."""
    config = QuantificationConfig(**_stored_quant_config(sample))
    payload = config.fingerprint_payload()
    payload["options"] = _canonicalize_json_value(
        {k: v for k, v in config.options.items() if k not in config.NON_FINGERPRINT_OPTIONS}
    )
    former = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert config.fingerprint() == former


@pytest.mark.parametrize("sample", OLD_LEDGERS)
def test_energy_range_selecting_same_channels_matches_old_quantification(sample):
    raw = _stored_quant_config(sample)
    old = QuantificationConfig(**raw)
    opts = raw["options"]
    sp_start, sp_end = old.fitted_channel_lims()
    # Energies of the first and last fitted channel select the same channels
    energy_range = [opts["det_ch_offset"] + opts["det_ch_width"] * sp_start,
                    opts["det_ch_offset"] + opts["det_ch_width"] * (sp_end - 1)]
    new_opts = {k: v for k, v in opts.items() if k != "spectrum_lims"}
    new = QuantificationConfig(**{**raw, "options": {**new_opts, "energy_range_keV": energy_range}})
    assert new.fitted_channel_lims() == (sp_start, sp_end)
    assert new.fingerprint() == old.fingerprint()


def test_options_need_exactly_one_fitted_range():
    raw = _stored_quant_config(OLD_LEDGERS[0])
    with pytest.raises(ValueError):
        QuantificationConfig(**{**raw, "options": {**raw["options"], "energy_range_keV": [0.1, 10.0]}})
    with pytest.raises(ValueError):
        QuantificationConfig(**{**raw, "options": {k: v for k, v in raw["options"].items() if k != "spectrum_lims"}})


# ----------------------------------------------------------------------------- skewed peak tail
def _former_tail(x, area, center, sigma, gamma):
    with np.errstate(all="ignore"):
        return area / (2 * gamma * sigma * np.exp(-1 / (2 * gamma**2))) \
            * np.exp((x - center) / (gamma * sigma)) \
            * erfc((x - center) / (np.sqrt(2) * sigma) + 1 / (np.sqrt(2) * gamma))


def test_tail_finite_far_above_low_energy_peak():
    # Cl Ll-like peak evaluated up to 14.5 keV: the former expression overflows to NaN
    x = np.linspace(0.1, 14.52, 1443)
    assert np.isnan(_former_tail(x, 1.0, 0.183, 0.0195, 1.004)).any()
    assert np.isfinite(Peaks_Model._gaussian_tail(x, 1.0, 0.183, 0.0195, 1.004)).all()


def test_tail_bitwise_unchanged_where_former_expression_finite():
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 14.52, 1443)
    for _ in range(300):
        args = (rng.uniform(1, 1e5), rng.uniform(0.1, 10), rng.uniform(0.015, 0.1), rng.uniform(0.3, 5))
        former = _former_tail(x, *args)
        finite = np.isfinite(former)
        assert np.array_equal(Peaks_Model._gaussian_tail(x, *args)[finite], former[finite])
