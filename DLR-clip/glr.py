"""
GLR: Gate Logit Refiner Module
Gate Logit Refiner Module

Core Innovations:
1. Channel Recalibration (CR) Block: Channel-wise recalibration, captures inter-class relationships
2. Asymmetric Gating: Asymmetric gate fusion, intelligently selects CLIP or CMA
3. Residual Correction: Residual refinement, protects original CLIP performance
"""

import torch
import torch.nn as nn


class ChannelRecalibrationBlock(nn.Module):
    """
    Channel Recalibration (CR) Block for Logits Feature
    Recalibrates importance across feature channels using normalization and gating.
    Captures inter-class dependencies more effectively than SE-Attention.
    """
    def __init__(self, channels, reduction=16):
        super(ChannelRecalibrationBlock, self).__init__()
        # Channel recalibration gate
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, Channels]
        # Channel recalibration gate
        scale = self.fc(x)
        
        # Apply recalibration
        return x * scale


class GateLogitRefiner(nn.Module):
    """
    GLR: Gate Logit Refiner (Improved version: asymmetric gating + residual enhancement)
    Asymmetric Gated Logit Refinement
    
    Core Innovations:
    1. Asymmetric Gating:
       - Gate network only sees CLIP features, not CMA features
       - CLIP is the "old professor" judging: "Do I know this problem?"
       - Prevents Gate from collapsing due to early CMA noise interference
    
    2. Residual Fusion:
       - Changed to f = f_clip + alpha * f_cma (instead of alpha*A + (1-alpha)*B)
       - CLIP always 100% preserved (baseline), CMA as increment
       - Even if alpha is small, CMA gradients can propagate through addition, not stuck by "either-or"
    
    3. Simplified Head:
       - Changed from 3-layer MLP to single layer Linear + zero initialization
       - 16-shot data is small, complex networks tend to overfit
       - Initialize to 0 to ensure delta_logits = 0, protect initial CLIP performance
    """
    
    def __init__(self, feature_dim=512, num_classes=1000, dropout=0.1, fixed_alpha=None):
        super(GateLogitRefiner, self).__init__()
        self.fixed_alpha = fixed_alpha  # None for adaptive, otherwise use fixed alpha value
        
        # 1. Feature Projection (Project Logits to Latent Space) - Keep simplified
        self.clip_project = nn.Linear(num_classes, feature_dim)
        self.cma_project = nn.Linear(num_classes, feature_dim)

        # 2. Attention Enhancement (Channel Recalibration Block) - Keep as core of deconfusion
        self.clip_cr = ChannelRecalibrationBlock(feature_dim, reduction=8)
        self.cma_cr = ChannelRecalibrationBlock(feature_dim, reduction=8)

        # 3. Asymmetric Gating - Core improvement
        # Input only has f_clip, does not see f_cma (prevent CMA noise interference)
        # Output is a weight alpha between 0~1, indicating "how much CMA help is needed"
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

        # 4. Correction Head (Residual Correction Head) - Changed to single layer + zero initialization
        # Since CR Block has done non-linear processing, use single layer Linear mapping here
        self.residual_head = nn.Linear(feature_dim, num_classes)
        # Key: Initialize to 0, ensure delta_logits = 0 at initialization
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        
        # 5. Auxiliary modules
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(feature_dim)
        
        # 6. Residual Scale - Changed to slightly larger initial value
        # Changed to 0.01 (instead of 1e-4), provides sufficient gradient signal for training
        self.res_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    # Note: Correction head is zero-initialized in __init__, no need for additional _init_weights method

    def forward(self, logits_clip, logits_cma):
        """
        Args:
            logits_clip: Original CLIP output logits [Batch, Num_Classes]
            logits_cma: CMA-enhanced output logits [Batch, Num_Classes]
        Returns:
            GLR_logits: Refined logits [Batch, Num_Classes]
        """
        
        # Save original dtype for final conversion
        original_dtype = logits_clip.dtype
        
        # Temporarily convert inputs and all module parameters to float32 for numerical stability
        with torch.autocast(device_type='cuda' if logits_clip.is_cuda else 'cpu', enabled=False):
            logits_clip = logits_clip.float()
            logits_cma = logits_cma.float()
            self.float()
            
            # --- Step 1: Project to latent space ---
            f_clip = self.act(self.clip_project(logits_clip))
            f_cma = self.act(self.cma_project(logits_cma))
            
            # --- Step 2: Channel Recalibration refinement ---
            f_clip = self.clip_cr(f_clip)
            f_cma = self.cma_cr(f_cma)
            
            # --- Step 3: Asymmetric Gating ---
            # Key improvement: Gate only sees CLIP features, determines "how much CMA help I need"
            # This makes Gate training very stable because CLIP features are frozen and high quality
            if self.fixed_alpha is not None:
                # Use fixed alpha value (ablation study mode)
                alpha = torch.full((f_clip.shape[0], 1), self.fixed_alpha, device=f_clip.device, dtype=f_clip.dtype)
            else:
                # Adaptively learn alpha (normal mode)
                alpha = self.gate_net(f_clip)  # [B, 1], input only has f_clip
            
            # --- Step 4: Residual Fusion ---
            # Core improvement: CLIP as primary, CMA as auxiliary
            # Logic: feature = CLIP feature + alpha * CMA increment feature
            # (instead of "life and death" competition of alpha*CLIP + (1-alpha)*CMA)
            f_fused = f_clip + alpha * f_cma
            
            # Add LayerNorm to stabilize distribution
            f_fused = self.ln(f_fused)
            
            # DEBUG: Print alpha statistics (disabled)
            # if not hasattr(self, '_debug_count'):
            #     self._debug_count = 0
            # self._debug_count += 1
            # if self._debug_count % 100 == 0:
            #     alpha_mean = alpha.mean().item()
            #     alpha_min = alpha.min().item()
            #     alpha_max = alpha.max().item()
            #     print(f"DEBUG GLR (不对称门控): fixed_alpha={self.fixed_alpha}, learned_alpha=[mean={alpha_mean:.4f}, min={alpha_min:.4f}, max={alpha_max:.4f}]")
            
            # --- Step 5: Generate correction term (Delta) ---
            # Single layer linear mapping, parameters already zero-initialized
            delta = self.residual_head(f_fused)
            
            # --- Step 6: Residual connection output ---
            # Final Logits = Original Logits + res_scale * correction term
            GLR_logits = logits_clip + self.res_scale * delta
        
        # Convert back to original dtype
        GLR_logits = GLR_logits.to(original_dtype)
        
        return GLR_logits


if __name__ == "__main__":
    batch_size = 4
    num_classes = 1000
    dim = 512
    
    # Simulate input
    logits_clip = torch.randn(batch_size, num_classes)
    logits_cma = torch.randn(batch_size, num_classes)
    
    # Instantiate model
    glr_model = GateLogitRefiner(feature_dim=dim, num_classes=num_classes)
    
    # Forward pass
    out = glr_model(logits_clip, logits_cma)
    
    print(f"Input shape: {logits_clip.shape}")
    print(f"Output shape: {out.shape}")
    
    # Check if gradients are normal (simple verification)
    loss = out.sum()
    loss.backward()
    print("Backward pass successful.")
