from glob import glob
from pathlib import Path
import click
import numpy as np
from tifffile import imread

from stardist.src.utils.config import read_json_config


def compute_iou(mask_pred, mask_gt):
    p = mask_pred > 0
    g = mask_gt > 0
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_containment_violation(pred_inner, pred_outer, gt_outer):
    """
    Computes Eq. 5 violation ratio:
    sum((1 - pred_outer) * pred_inner) / sum(1 - gt_outer)
    """
    p1 = (pred_inner > 0).astype(np.float64)
    p2 = (pred_outer > 0).astype(np.float64)
    g2 = (gt_outer > 0).astype(np.float64)

    p2_perp = 1.0 - p2
    g2_perp = 1.0 - g2

    viol_num = np.sum(p2_perp * p1)
    bg_denom = max(1.0, np.sum(g2_perp))
    return viol_num / bg_denom


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    parameters = config.get("parameters", {})
    ext = parameters.get("extension", ".tif")

    test_gt_y1 = sorted(glob(os.path.join(config["test"]["mask1_dir"], f"*{ext}")))
    test_gt_y2 = sorted(glob(os.path.join(config["test"]["mask2_dir"], f"*{ext}")))
    pred_y1 = sorted(glob(os.path.join(config["test_pred"]["mask1_dir"], f"*{ext}")))
    pred_y2 = sorted(glob(os.path.join(config["test_pred"]["mask2_dir"], f"*{ext}")))

    assert len(test_gt_y1) == len(pred_y1), f"Mismatch in mask1 files: {len(test_gt_y1)} vs {len(pred_y1)}"
    assert len(test_gt_y2) == len(pred_y2), f"Mismatch in mask2 files: {len(test_gt_y2)} vs {len(pred_y2)}"

    ious_y1 = []
    ious_y2 = []
    violations = []

    for g1_p, g2_p, p1_p, p2_p in zip(test_gt_y1, test_gt_y2, pred_y1, pred_y2):
        g1 = imread(g1_p)
        g2 = imread(g2_p)
        p1 = imread(p1_p)
        p2 = imread(p2_p)

        ious_y1.append(compute_iou(p1, g1))
        ious_y2.append(compute_iou(p2, g2))
        violations.append(compute_containment_violation(p1, p2, g2))

    print("================ Evaluation Results ================")
    print(f"Number of test images: {len(test_gt_y1)}")
    print(f"Mean IoU (Inner - Nuclei/Plaques): {np.mean(ious_y1):.4f} +/- {np.std(ious_y1):.4f}")
    print(f"Mean IoU (Outer - Cytoplasm/Wells): {np.mean(ious_y2):.4f} +/- {np.std(ious_y2):.4f}")
    print(f"Mean Containment Violation Ratio:  {np.mean(violations):.6f} +/- {np.std(violations):.6f}")
    print("====================================================")


if __name__ == "__main__":
    import os
    main()

