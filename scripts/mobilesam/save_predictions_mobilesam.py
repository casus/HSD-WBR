import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from glob import glob
from pathlib import Path
import click
import numpy as np
import torch
import torch.nn.functional as F
from tifffile import imwrite
from torch.utils.data import DataLoader

from stardist.src.utils.config import read_json_config
from scripts.mobilesam.dataset import NestedBiomedicalDataset


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
def main(config_file_path):
    config = read_json_config(config_file_path)
    parameters = config.get("parameters", {})

    device = torch.device(
        "mps"
        if (parameters.get("use_gpu", True) and torch.backends.mps.is_available())
        else ("cuda" if (parameters.get("use_gpu", True) and torch.cuda.is_available()) else "cpu")
    )
    print(f"Using device for inference: {device}")

    try:
        from mobile_sam import sam_model_registry
    except ImportError:
        raise ImportError(
            "MobileSAM is not installed. Please install it using: "
            "pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
        )

    # Load model checkpoint
    checkpoint_path = parameters.get("checkpoint_path", "weights/mobile_sam.pt")
    model = sam_model_registry["vit_t"](checkpoint=None)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: {checkpoint_path} not found. Running with initial weights.")

    model.to(device)
    model.eval()

    # Output directory
    results_dir = config.get("results_dir", "examples/mobilesam/models/")
    log_dir = parameters.get("log_dir", "predictions")
    pred_mask1_dir = Path(results_dir) / log_dir / "mask1"
    pred_mask2_dir = Path(results_dir) / log_dir / "mask2"
    pred_mask1_dir.mkdir(parents=True, exist_ok=True)
    pred_mask2_dir.mkdir(parents=True, exist_ok=True)

    ext = parameters.get("extension", ".tif")
    test_imgs = sorted(glob(os.path.join(config["test"]["img_dir"], f"*{ext}")))
    test_m1 = sorted(glob(os.path.join(config["test"]["mask1_dir"], f"*{ext}")))
    test_m2 = sorted(glob(os.path.join(config["test"]["mask2_dir"], f"*{ext}")))

    if len(test_imgs) == 0:
        print(f"No test images found in {config['test']['img_dir']}.")
        return

    test_dataset = NestedBiomedicalDataset(test_imgs, test_m1, test_m2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Generating predictions for {len(test_dataset)} test samples...")
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            boxes = batch["boxes"].squeeze(0).to(device)
            orig_h = int(batch["orig_size"][0][0])
            orig_w = int(batch["orig_size"][1][0])
            fname = batch["file_name"][0]

            image_embeddings = model.image_encoder(image)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            # Token 0 -> Inner (nuclei / plaques)
            # Token 2 -> Outer (cytoplasm / wells)
            pred_inner = torch.amax(low_res_masks[:, 0:1, :, :], dim=0, keepdim=True)
            pred_outer = torch.amax(low_res_masks[:, 2:3, :, :], dim=0, keepdim=True)

            # Upsample back to original image resolution
            mask1_full = F.interpolate(pred_inner, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            mask2_full = F.interpolate(pred_outer, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

            bin_mask1 = (mask1_full.squeeze().cpu().numpy() > 0.0).astype(np.uint32)
            bin_mask2 = (mask2_full.squeeze().cpu().numpy() > 0.0).astype(np.uint32)

            # Save as TIFF compatible with JTPR evaluation scripts
            save_name1 = pred_mask1_dir / f"{fname}_img_pred{ext}"
            save_name2 = pred_mask2_dir / f"{fname}_img_pred{ext}"

            imwrite(str(save_name1), bin_mask1)
            imwrite(str(save_name2), bin_mask2)

    print(f"Predictions saved to:\n - {pred_mask1_dir}\n - {pred_mask2_dir}")


if __name__ == "__main__":
    main()

