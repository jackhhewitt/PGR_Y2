import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AlphaFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = torch.clamp(pt, min=1e-8, max=1.0-1e-8)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * (1 - pt)**self.gamma * torch.log(pt)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE with logits (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )
        # Probabilities (needed for focal term)
        probs = torch.sigmoid(logits)
        # p_t: probability of the true label
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # Focal loss scaling
        focal_term = (1 - p_t) ** self.gamma
        # Alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_term * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
class FocalLossSigmoidMargin(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLossSigmoidMargin, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # targets are -1 or 1
        # margin = y * f(x)
        margin = targets * logits
        
        # p_t for margin-based is sigmoid of the margin
        p_t = torch.sigmoid(margin)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Standard Margin-based Binary Loss (Soft Margin Loss)
        # log(1 + exp(-margin))
        bce_loss = F.soft_margin_loss(logits, targets, reduction='none')
        
        # Alpha balancing: needs to map -1 to (1-alpha) and 1 to alpha
        # Formula: 0.5 * ( (1 + y)*alpha + (1 - y)*(1 - alpha) )
        alpha_t = 0.5 * ((1 + targets) * self.alpha + (1 - targets) * (1 - self.alpha))
        
        loss = alpha_t * focal_term * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum() if self.reduction == "sum" else loss