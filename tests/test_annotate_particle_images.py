#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw particle images saved during acquisition, and their annotation from the ledger."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import autoemx.utils.constants as cnst
from autoemx.config.ledger_io import load_sample_ledger
from autoemx.config.ledger_schemas import AcquisitionDetails, ParticleInfo, SpotCoordinates
from autoemx.config.runtime_configs import MeasurementConfig
from autoemx.core.em_runtime.image_utilities import save_frame_image
from autoemx.runners.annotate_particle_images import annotate_particle_images
from ci_paths import INPUTS_DIR, WULFENITE_MINI_ID

SPOTS = [(60, 50), (140, 90)]
PIXEL_SIZE_UM = 0.05


def _is_red(pixels: np.ndarray) -> np.ndarray:
    return (pixels[..., 0] > 200) & (pixels[..., 1] < 60) & (pixels[..., 2] < 60)


def _make_sample(tmp_path: Path, pixel_size_in_ledger: bool) -> Path:
    sample_dir = tmp_path / WULFENITE_MINI_ID
    shutil.copytree(INPUTS_DIR / WULFENITE_MINI_ID, sample_dir)
    ledger_path = sample_dir / f"{cnst.LEDGER_FILENAME}{cnst.LEDGER_FILEEXT}"
    ledger = load_sample_ledger(str(ledger_path))
    for spectrum, xy in zip(ledger.spectra, SPOTS):
        spectrum.acquisition_details = AcquisitionDetails(
            frame_id="0", particle_id=3,
            spot_coordinates=SpotCoordinates(pixel_coordinates=xy),
        )
    ledger.particles = [ParticleInfo(
        id=3, frame_id="0", image_pixel_size_um=PIXEL_SIZE_UM if pixel_size_in_ledger else None,
    )]
    ledger.to_json_file(ledger_path)

    # Raw particle image, saved as during acquisition with annotate_particle_images=False
    images_dir = sample_dir / cnst.IMAGES_DIR
    images_dir.mkdir()
    frame = np.full((150, 200), 90, dtype=np.uint8)
    save_frame_image(
        frame, PIXEL_SIZE_UM, 200, 150, WULFENITE_MINI_ID, ledger.configs.microscope_cfg,
        f"{WULFENITE_MINI_ID}_par3_fr0_xyspots", str(images_dir),
        im_annotations=None, scalebar=False, image_extension="png", save_raw_image=True,
    )
    return sample_dir


def test_raw_particle_images_are_the_default():
    assert MeasurementConfig().annotate_particle_images is False


def test_raw_image_saved_with_pixel_size_metadata(tmp_path: Path):
    sample_dir = _make_sample(tmp_path, pixel_size_in_ledger=True)
    images = sorted((sample_dir / cnst.IMAGES_DIR).iterdir())
    # No duplicate "_raw" copy when nothing is drawn on the image
    assert [p.name for p in images] == [f"{WULFENITE_MINI_ID}_par3_fr0_xyspots.png"]
    with Image.open(images[0]) as im:
        assert json.loads(im.info["Description"])["pixel_size_um"] == PIXEL_SIZE_UM
        assert not _is_red(np.array(im.convert("RGB"))).any()


def test_annotate_particle_images(tmp_path: Path):
    for pixel_size_in_ledger in (True, False):  # scale from the ledger, or from the image metadata
        sample_dir = _make_sample(tmp_path / str(pixel_size_in_ledger), pixel_size_in_ledger)
        written = annotate_particle_images(sample_path=str(sample_dir))
        assert len(written) == 1
        out = Path(written[0])
        assert out.parent == sample_dir / cnst.IMAGES_DIR / cnst.ANNOTATED_IMAGES_SUBDIR
        annotated = np.array(Image.open(out).convert("RGB"))
        red = _is_red(annotated)
        for x, y in SPOTS:
            assert red[y, x]  # dot centred on each spot
        assert (annotated.min(axis=2) > 250).sum() > 50  # white scale bar
        # The original image is not modified
        original = np.array(Image.open(sample_dir / cnst.IMAGES_DIR / out.name).convert("RGB"))
        assert not _is_red(original).any()
