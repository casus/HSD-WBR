import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
from glob import glob
from pathlib import Path
import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from stardist.src.utils.config import read_json_config
from scripts.mobilesam.dataset import NestedBiomedicalDataset
from scripts.mobilesam.losses import WBRLoss, standard_sam_mask_loss


@click.command()
@click.argument("config_file_path", type=click.Path(exists=True))
@click.option("--seed", type=int, default=None, help="Random seed override")
def main(config_file_path, seed):
    config = read_json_config(config_file_path)
    parameters = config.get("parameters", {})

    chosen_seed = seed if seed is not None else int(parameters.get("seed", 11))
    np.random.seed(chosen_seed)
    torch.manual_seed(chosen_seed)

    # 1. Hardware device setup
    use_gpu = parameters.get("use_gpu", True)
    if use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} (seed: {chosen_seed})")

    # 2. Directory and checkpoint setup
    running_time = time.strftime("%b-%d-%Y_%H-%M")
    results_dir = config.get("results_dir", "examples/mobilesam/models/")
    model_dir = Path(results_dir) / "mobilesam" / running_time
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint and logs will be saved to: {model_dir}")

    # 3. Model instantiation
    try:
        from mobile_sam import sam_model_registry
    except ImportError:
        raise ImportError(
            "MobileSAM is not installed. Please install it using: "
            "pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
        )

    checkpoint_path = parameters.get("checkpoint_path", "weights/mobile_sam.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Attempting to initialize model without weights.")
        checkpoint_path = None

    model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    model.to(device)

    # Freeze image encoder if requested (saves memory on M1 Max)
    freeze_encoder = parameters.get("freeze_image_encoder", True)
    if freeze_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen.")

    # 4. Data loading
    ext = parameters.get("extension", ".tif")
    train_imgs = sorted(glob(os.path.join(config["train"]["img_dir"], f"*{ext}")))
    train_m1 = sorted(glob(os.path.join(config["train"]["mask1_dir"], f"*{ext}")))
    train_m2 = sorted(glob(os.path.join(config["train"]["mask2_dir"], f"*{ext}")))

    if len(train_imgs) == 0:
        print(f"No training data found in {config['train']['img_dir']}. Exiting early.")
        return

    train_dataset = NestedBiomedicalDataset(train_imgs, train_m1, train_m2)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # 5. Optimizers and loss criteria
    lr = float(parameters.get("lr", 1e-4))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    train_wbr = parameters.get("train_wbr_loss", False)
    wbr_weight = float(parameters.get("wbr_loss_weight", 1.0))
    wbr_eps = float(parameters.get("wbr_eps", 1e-3))
    wbr_criterion = WBRLoss(eps=wbr_eps)

    epochs = int(parameters.get("epochs", 50))
    print(f"Training for {epochs} epochs | WBR enabled: {train_wbr} (weight={wbr_weight}, eps={wbr_eps})")

    # 6. Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_wbr_viol = 0.0
        step_count = 0

        for batch in train_loader:
            image = batch["image"].to(device)  # [1, 3, 1024, 1024]
            boxes = batch["boxes"].squeeze(0).to(device)  # [N, 4]
            gt_y1 = batch["y1"].to(device)  # [1, 1, 256, 256]
            gt_y2 = batch["y2"].to(device)  # [1, 1, 256, 256]

            optimizer.zero_grad()

            # Forward pass: extract shared image embeddings once
            with torch.set_grad_enabled(not freeze_encoder):
                image_embeddings = model.image_encoder(image)

            # Prompt encoder (boxes)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )

            # Mask decoder with multimask_output=True (yields 3 masks: Token 0, 1, 2)
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            # Map hierarchical tokens:
            # Token 0 -> Subpart / Inner object (Nuclei / Plaques)
            # Token 2 -> Whole / Outer object (Cytoplasm / Wells)
            pred_inner_inst = low_res_masks[:, 0:1, :, :]  # [N, 1, 256, 256]
            pred_outer_inst = low_res_masks[:, 2:3, :, :]  # [N, 1, 256, 256]

            # Differentiable union across instances for global image representation
            pred_inner_semantic = torch.amax(pred_inner_inst, dim=0, keepdim=True)
            pred_outer_semantic = torch.amax(pred_outer_inst, dim=0, keepdim=True)

            # Standard SAM losses (Focal + Dice)
            loss_inner = standard_sam_mask_loss(pred_inner_semantic, gt_y1)
            loss_outer = standard_sam_mask_loss(pred_outer_semantic, gt_y2)
            total_loss = loss_inner + loss_outer

            # WBR penalty computation (computed in float32)
            if train_wbr and wbr_weight > 0:
                lambda_wbr, viol_ratio = wbr_criterion(
                    pred_inner_semantic,
                    pred_outer_semantic,
                    gt_y2,
                )
                total_loss = total_loss + (wbr_weight * lambda_wbr)
                epoch_wbr_viol += viol_ratio.item()

            total_loss.backward()
            # Gradient clipping to prevent O(1/eps^2) spikes
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            step_count += 1

        avg_loss = epoch_loss / max(1, step_count)
        avg_viol = epoch_wbr_viol / max(1, step_count)
        print(f"Epoch {epoch:3d}/{epochs:3d} | Loss: {avg_loss:.4f} | WBR Violation Ratio: {avg_viol:.4f}")

    # 7. Save trained weights
    save_path = model_dir / "mobilesam_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

