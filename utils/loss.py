# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes=1, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Standard Dice Loss.
        targets: (B, H, W) with integer values [0, n_classes-1]
        """
        # --- Binary Mode ---
        if self.n_classes == 1:
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)
            intersection = (probs_flat * targets_flat).sum()
            dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
            return 1 - dice

        # --- Multi-class Mode ---
        else:
            probs = torch.softmax(logits, dim=1)
            
            # One-Hot Encoding
            targets_one_hot = F.one_hot(targets, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
            
            # Flatten
            probs_flat = probs.contiguous().view(self.n_classes, -1)
            targets_flat = targets_one_hot.contiguous().view(self.n_classes, -1)
            
            intersection = (probs_flat * targets_flat).sum(dim=1)
            union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
            
            dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
            
            return 1 - dice_per_class.mean()

class SegmentationLoss(nn.Module):
    def __init__(self, n_classes=1, weight_dice=0.5, weight_ce=0.5, ignore_index=None):
        """
        Args:
            n_classes (int): Number of classes.
            weight_dice (float): Weight for Dice Loss.
            weight_ce (float): Weight for CrossEntropy/BCE Loss.
            ignore_index (int, optional): Index to ignore in CrossEntropy. Default: None.
        """
        super(SegmentationLoss, self).__init__()
        self.n_classes = n_classes
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        
        self.dice_loss = DiceLoss(n_classes=n_classes)
        
        if n_classes == 1:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            # If config.IGNORE_INDEX is None (LaPa)，CrossEntropy will calculate all pixels
            # If config.IGNORE_INDEX is 255 (VOC)，CrossEntropy will ignore it
            if ignore_index is not None:
                self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
            else:
                self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        if self.n_classes == 1:
            loss_ce = self.ce_loss(logits, targets.float())
        else:
            loss_ce = self.ce_loss(logits, targets.long())
            
        loss_dice = self.dice_loss(logits, targets)
        
        return (self.weight_ce * loss_ce) + (self.weight_dice * loss_dice)