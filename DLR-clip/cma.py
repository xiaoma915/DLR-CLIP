"""
CMA (Cross-Modal Adapter) Module
Lightweight adapters for multi-modal learning in CLIP
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import clip_dlr as clip


class CMAAdapter(nn.Module):
    """
    CMA Adapter: Lightweight layer-wise adapters for text and visual transformers
    """
    def __init__(self, d_model, n_layers, l_start, l_end, mid_dim, dtype):
        super().__init__()
        # Create adapters for all layers, but only initialize specific ones
        self.adapters = nn.ModuleList([None] * (n_layers + 1))
        for i in range(l_start, min(l_end + 1, n_layers + 1)):  # Make sure we don't exceed the number of layers
            if mid_dim == d_model:
                adapter = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
            self.adapters[i] = adapter
        # Initialize weights for initialized adapters only
        for adapter in self.adapters:
            if adapter is not None:
                for m in adapter.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                        nn.init.constant_(m.bias, 0)
        # Handle half precision
        if dtype == torch.float16:
            for adapter in self.adapters:
                if adapter is not None:
                    for m in adapter.modules():
                        m.half()

    def forward(self, x, layer_index):
        if layer_index < len(self.adapters) and self.adapters[layer_index] is not None:
            adapter = self.adapters[layer_index]
            # Apply adapter transformation
            if hasattr(adapter, 'down'):  # Complex adapter (down/up)
                residual = x
                x = adapter.down(x)
                x = adapter.up(x)
                x = residual + x  # 标准残差连接
            else:  # Simple adapter
                residual = x
                x = adapter(x)
                x = residual + x  # 标准残差连接
            return x
        return None


class CMAAdapterLearner(nn.Module):
    """
    CMA Adapter Learner: Manages text and visual adapters for class-wise learning
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0] if hasattr(cfg, 'INPUT') and hasattr(cfg.INPUT, 'SIZE') else clip_imsize
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self._build_text_embedding(cfg, classnames, clip_model)

        # Get multi-modal adapter configuration
        adapter_start = getattr(cfg.TRAINER.CMAADAPTER, 'ADAPTER_START', 5)
        adapter_end = getattr(cfg.TRAINER.CMAADAPTER, 'ADAPTER_END', 12)
        adapter_dim = getattr(cfg.TRAINER.CMAADAPTER, 'ADAPTER_DIM', 32)
        self.adapter_scale = float(getattr(cfg.TRAINER.CMAADAPTER, 'ADAPTER_SCALE', 1.0))

        # For ViT-B/16, we have 12 transformer layers (0-11)
        # For text transformer
        self.text_adapter = CMAAdapter(
            clip_model.ln_final.weight.shape[0],  # d_model
            len(clip_model.transformer.resblocks) - 1,  # n_layers (0-indexed)
            adapter_start,
            adapter_end,
            adapter_dim,
            clip_model.dtype
        )

        # For visual transformer
        self.visual_adapter = CMAAdapter(
            clip_model.visual.ln_post.weight.shape[0],  # d_model
            len(clip_model.visual.transformer.resblocks) - 1,  # n_layers (0-indexed)
            adapter_start,
            adapter_end,
            adapter_dim,
            clip_model.dtype
        )

        # Shared adapters
        self.shared_adapter = CMAAdapter(
            adapter_dim,
            max(len(clip_model.visual.transformer.resblocks), len(clip_model.transformer.resblocks)) - 1,  # Use max of both
            adapter_start,
            adapter_end,
            adapter_dim,
            clip_model.dtype
        )

    def _build_text_embedding(self, cfg, classnames, clip_model):
        dtype = clip_model.dtype
        text_ctx_init = getattr(cfg.TRAINER.CMAADAPTER, 'TEXT_CTX_INIT', "a photo of a")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [text_ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            # Move tokenized_prompts to the same device as clip_model first
            if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'conv1'):
                device = clip_model.visual.conv1.weight.device
            else:
                device = next(clip_model.parameters()).device
            tokenized_prompts = tokenized_prompts.to(device)
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Register buffers (they're already on the correct device)
        self.register_buffer("token_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def forward(self):
        embedding = self.token_embedding
        # Apply initial adapter if exists (index 0)
        if self.text_adapter.adapters[0] is not None:
            token_embedding = self.text_adapter.adapters[0].down(embedding)
            if self.shared_adapter.adapters[0] is not None:
                token_embedding = self.shared_adapter.adapters[0](token_embedding)
            token_embedding = self.text_adapter.adapters[0].up(embedding)
            embedding = embedding + self.adapter_scale * token_embedding
        return embedding, self.text_adapter, self.visual_adapter, self.shared_adapter, self.adapter_scale
