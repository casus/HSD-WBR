import torch
import torch.nn as nn
import torch.nn.functional as F


class WBRLoss(nn.Module):
    """
    Differentiable Within Boundary Regularisation (WBR) Penalty (Eq. 5 in paper).

    Lambda = (1 + eps - sum(y2_hat_perp * y1_hat) / sum(y2_gt_perp))^(-1)

    where:
      - y1_hat: predicted inner object probability (e.g. nucleus or plaque), shape [B, 1, H, W]
      - y2_hat: predicted outer object probability (e.g. cytoplasm or well), shape [B, 1, H, W]
      - y2_hat_perp: 1 - y2_hat (predicted outer background)
      - y2_gt_perp: 1 - gt_outer (ground-truth outer background)
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred_inner_logits, pred_outer_logits, gt_outer_semantic):
        # Always compute in float32 for Apple Silicon MPS numerical stability
        y1_hat = torch.sigmoid(pred_inner_logits.float())
        y2_hat = torch.sigmoid(pred_outer_logits.float())

        y2_hat_perp = 1.0 - y2_hat
        y2_gt_perp = 1.0 - gt_outer_semantic.float()

        violation_num = torch.sum(y2_hat_perp * y1_hat)
        bg_area_denom = torch.sum(y2_gt_perp).clamp(min=1.0)

        violation_ratio = (violation_num / bg_area_denom).clamp(0.0, 1.0)
        lambda_wbr = 1.0 / (1.0 + self.eps - violation_ratio)
        return lambda_wbr, violation_ratio


def dice_loss(pred_logits, gt_targets, smooth=1.0):
    """
    Soft Dice loss computed over spatial dimensions [H, W].
    """
    probs = torch.sigmoid(pred_logits.float())
    gt = gt_targets.float()
    dims = (1, 2, 3)
    intersection = 2.0 * torch.sum(probs * gt, dims) + smooth
    union = torch.sum(probs, dims) + torch.sum(gt, dims) + smooth
    return (1.0 - (intersection / union)).mean()


def focal_loss(pred_logits, gt_targets, alpha=0.25, gamma=2.0):
    """
    Sigmoid Focal loss for binary mask prediction.
    """
    p = torch.sigmoid(pred_logits.float())
    gt = gt_targets.float()
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits.float(), gt, reduction="none")
    p_t = p * gt + (1.0 - p) * (1.0 - gt)
    alpha_factor = alpha * gt + (1.0 - alpha) * (1.0 - gt)
    modulating_factor = (1.0 - p_t) ** gamma
    return (alpha_factor * modulating_factor * ce_loss).mean()


def standard_sam_mask_loss(pred_logits, gt_masks):
    """
    Linear combination of focal loss and dice loss standard in SAM mask supervision.
    """
    l_focal = focal_loss(pred_logits, gt_masks)
    l_dice = dice_loss(pred_logits, gt_masks)
    return 20.0 * l_focal + 1.0 * l_dice

