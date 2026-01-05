import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def distill(y_s, y_t, T=1.0, alpha=0.0):
    """
    Compute smoothed knowledge distillation loss
    
    Args:
        y_s: Student logits (batch_size, num_classes)
        y_t: Teacher logits (batch_size, num_classes)
        T: Temperature for softening
        alpha: Alpha smoothing parameter for mixing teacher outputs with label smoothing
               When alpha > 0, blend teacher probabilities with uniform distribution
        
    Returns:
        Knowledge distillation loss
    """
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = F.softmax(y_t / T, dim=1)
    
    # If alpha > 0, blend teacher probabilities with label smoothing
    if alpha > 0:
        # Create a uniform distribution for label smoothing
        batch_size = y_t.shape[0]
        num_classes = y_t.shape[1]
        
        # Uniform distribution (for label smoothing effect)
        uniform_dist = torch.ones(batch_size, num_classes, device=y_t.device, dtype=p_t.dtype) / num_classes
        
        # Blend: alpha * uniform + (1 - alpha) * teacher
        # This smooths the teacher targets towards uniform distribution
        p_t = alpha * uniform_dist + (1.0 - alpha) * p_t
    
    loss = F.kl_div(p_s, p_t, reduction="batchmean") * (T ** 2)
    return loss


class SKD(nn.Module):
    """
    SKD: Smoothed Knowledge Distillation module
    Uses the original CLIP model logits (without CMA/GLR) as teacher targets,
    and distills the student model (with CMA + GLR modifications) using these targets.
    
    Key insight: The teacher is the pure CLIP model without any modifications (CMA/GLR),
    while the student is the full modified model. This creates a meaningful knowledge distillation objective.
    """
    
    def __init__(self, student_model, device, temperature=1.0, alpha_smoothing=0.0):
        """
        Initialize SKD module
        
        Args:
            student_model: The student model being trained
            device: Device to run on
            temperature: Temperature for softening probability distributions
            alpha_smoothing: Alpha smoothing parameter for combining ground truth with teacher outputs
        """
        super(SKD, self).__init__()
        self.device = device
        self.T = temperature
        self.alpha = alpha_smoothing
    
    def forward(self, student_logits, teacher_logits):
        """
        Forward pass with knowledge distillation
        
        Args:
            student_logits: Student model logits (final_logits from CMA+GLR forward)
            teacher_logits: Teacher model logits (pure CLIP logits, computed separately)
            
        Returns:
            Knowledge distillation loss
        """
        # Compute knowledge distillation loss
        # student_logits = final_logits (from CMA+GLR)
        # teacher_logits = pure CLIP logits (frozen, computed in training loop)
        kd_loss = distill(student_logits, teacher_logits, self.T, self.alpha)
        
        return kd_loss
