import json
import math
from pathlib import Path
import numpy as np
import tifffile as tiff
import random
from typing import Union

import click
from PIL import Image, ImageDraw

from stardist.src.utils.config import read_json_config

def load_mask(path, threshold=0):
    """Load a TIFF and return a boolean mask."""
    img = tiff.imread(str(path))

    # If RGB/RGBA or multi-channel, collapse to 2D (max over channels)
    if img.ndim == 3:
        # common cases: (H, W, C) or (C, H, W)
        if img.shape[-1] in (3, 4):
            img = img.max(axis=-1)
        elif img.shape[0] in (3, 4):
            img = img.max(axis=0)

    # If multi-page stack (Z, H, W), choose a strategy:
    # Here: OR over slices to get a single 2D mask.
    if img.ndim == 3:
        img = img.max(axis=0)

    # Binarize
    return img > threshold

def overlap_metrics(mask_a, mask_b):
    """Return overlap metrics for two boolean masks."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}")

    inter = mask_a & mask_b

    a = int(mask_a.sum())
    inter_n = int(inter.sum())

    overlap = inter_n / a if a else np.nan

    return overlap

@click.command()
@click.argument('config_file_path', type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)

    current_dir	= config['project_result_dir']
    if current_dir:
        current_dir = Path(current_dir)
    else:
        current_dir = Path(__file__).parent

    print(current_dir)
    parameters = config['parameters']

    circle_mask_dir = Path(parameters['circle_mask_dir'])
    star_mask_dir = Path(parameters['star_mask_dir'])

    circle_paths = sorted(circle_mask_dir.glob("*."+parameters["target_type"]))
    star_paths = sorted(star_mask_dir.glob("*."+parameters["target_type"]))

    circles = [load_mask(p, 0) for p in circle_paths]
    stars = [load_mask(p, 0) for p in star_paths]

    mean_overlap = np.mean([overlap_metrics(c,s) for c,s in zip(circles, stars)])
    print(f"Mean overlap between circle and star masks: {mean_overlap:.4f}")


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter