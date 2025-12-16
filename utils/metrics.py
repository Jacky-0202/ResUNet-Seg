# utils/metrics.py

import torch
import torch.nn.functional as F

def calculate_dice(logits, targets, n_classes=1, ignore_index=255):
    """
    Calculate Dice Score for both Binary and Multi-class tasks.
    
    Args:
        logits (torch.Tensor): Model output [B, C, H, W] or [B, 1, H, W]
        targets (torch.Tensor): Ground truth [B, H, W] (indices) or [B, 1, H, W]
        n_classes (int): Number of classes.
        ignore_index (int): Class index to ignore (for multi-class).
        
    Returns:
        float: Average Dice Score.
    """
    smooth = 1e-6
    
    # --- Binary Mode ---
    if n_classes == 1:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Ensure targets are float
        targets = targets.float()
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    # --- Multi-class Mode ---
    else:
        # Get predictions (Argmax)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1) # [B, H, W]
        
        dice_sum = 0.0
        valid_classes = 0
        
        for c in range(n_classes):
            # Skip the ignore_index class if strictly required, 
            # usually we just ignore pixels in the mask.
            
            # Create binary masks for the current class c
            pred_binary = (preds == c)
            target_binary = (targets == c)
            
            # Handle ignore_index in targets (mask out void pixels)
            if ignore_index is not None:
                valid_mask = (targets != ignore_index)
                pred_binary = pred_binary & valid_mask
                target_binary = target_binary & valid_mask

            intersection = (pred_binary & target_binary).sum().float()
            union = pred_binary.sum().float() + target_binary.sum().float()
            
            if union == 0:
                # If both are empty, score is 1. If prediction is empty but target is not, score is 0.
                if target_binary.sum() == 0 and pred_binary.sum() == 0:
                    score = 1.0
                else:
                    score = 0.0
            else:
                score = (2. * intersection + smooth) / (union + smooth)
            
            dice_sum += score
            valid_classes += 1
            
        return (dice_sum / valid_classes).item()