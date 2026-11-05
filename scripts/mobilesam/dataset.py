import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tifffile import imread
from pathlib import Path


class NestedBiomedicalDataset(Dataset):
    """
    Dataset loader for nested biomedical microscopy data (e.g. HeLaCytoNuc, VACVPlaque).
    - Image: raw micrograph (.tif)
    - Mask 1 (inner): nuclei or viral plaques
    - Mask 2 (outer): cytoplasm or well boundary
    """

    def __init__(self, img_paths, mask1_paths, mask2_paths, target_size=1024):
        self.img_paths = sorted(img_paths)
        self.mask1_paths = sorted(mask1_paths)
        self.mask2_paths = sorted(mask2_paths)
        self.target_size = target_size

        assert len(self.img_paths) == len(self.mask1_paths) == len(self.mask2_paths), (
            f"Counts mismatch: {len(self.img_paths)} images, "
            f"{len(self.mask1_paths)} mask1, {len(self.mask2_paths)} mask2"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load TIFFs explicitly as float32 to prevent float64 MPS conversion errors
        img = imread(self.img_paths[idx]).astype(np.float32)
        y1 = imread(self.mask1_paths[idx]).astype(np.float32)
        y2 = imread(self.mask2_paths[idx]).astype(np.float32)

        # 2. Normalize image to [0, 255] and expand to 3 channels (RGB)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)

        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] > 3:
            img = img[..., :3]

        orig_h, orig_w = img.shape[:2]

        # 3. Scale longest side to target_size
        scale = self.target_size / max(orig_h, orig_w)
        new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        img_resized = F.interpolate(img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Pad to [target_size, target_size]
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        img_padded = F.pad(img_resized, (0, pad_w, 0, pad_h), value=0.0).squeeze(0)  # [3, 1024, 1024]

        # Standard SAM mean/std pixel normalization
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        img_norm = (img_padded - pixel_mean) / pixel_std

        # 4. Resize ground truth masks (binary semantic representation)
        y1_tensor = torch.from_numpy((y1 > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        y2_tensor = torch.from_numpy((y2 > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)

        # Target mask resolution: standard SAM low_res_masks are 256x256 (64x64 embedding upsampled 4x)
        mask_res = 256
        y1_mask = F.interpolate(y1_tensor, size=(mask_res, mask_res), mode="nearest").squeeze(0)
        y2_mask = F.interpolate(y2_tensor, size=(mask_res, mask_res), mode="nearest").squeeze(0)

        # 5. Extract bounding boxes from outer objects (e.g. cells or wells)
        boxes = self._extract_boxes(y2, scale, pad=(0, 0))

        return {
            "image": img_norm,
            "boxes": boxes,
            "y1": y1_mask,  # [1, 256, 256]
            "y2": y2_mask,  # [1, 256, 256]
            "orig_size": (orig_h, orig_w),
            "file_name": Path(self.img_paths[idx]).stem,
        }

    @staticmethod
    def _extract_boxes(instance_mask, scale, pad=(0, 0)):
        labels = np.unique(instance_mask)
        labels = labels[labels > 0]
        boxes = []
        for lbl in labels:
            ys, xs = np.where(instance_mask == lbl)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min = xs.min() * scale
            y_min = ys.min() * scale
            x_max = xs.max() * scale
            y_max = ys.max() * scale
            boxes.append([x_min, y_min, x_max, y_max])

        if len(boxes) == 0:
            # Fallback to whole frame bounding box if no instances
            h, w = instance_mask.shape[:2]
            boxes = [[0.0, 0.0, float(w * scale), float(h * scale)]]

        return torch.tensor(boxes, dtype=torch.float32)

