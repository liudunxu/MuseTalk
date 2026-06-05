"""Vendored CodeFormer model (inference only).

This is a minimal vendoring of the CodeFormer network from
https://github.com/sczhou/CodeFormer (NTU S-Lab License 1.0),
specifically ``basicsr/archs/codeformer_arch.py``. The training-time
plumbing (``ARCH_REGISTRY.register()`` decorator, ``get_root_logger``,
positional encoding with masks, the ``TransformerSALayer`` constructor
defaults) is preserved exactly so that the published
``codeformer.pth`` checkpoint loads without key remapping.

What we strip:
  * ``ARCH_REGISTRY`` -- we instantiate the class by name directly.
  * ``get_root_logger`` -- vendored module is silent at import time.

What we keep:
  * ``calc_mean_std`` / ``adaptive_instance_normalization``
  * ``PositionEmbeddingSine``
  * ``_get_activation_fn`` / ``TransformerSALayer``
  * ``Fuse_sft_block``
  * ``CodeFormer`` (extends the vendored ``VQAutoEncoder``)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .vqgan_arch import ResBlock, VQAutoEncoder


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, "The input feature should be 4D tensor."
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        return dec_feat + residual


class CodeFormer(VQAutoEncoder):
    """CodeFormer blind face restoration network.

    Constructor signature is the upstream one (parameters keyed by the
    released checkpoint). The inference entry point is ``forward(x, w)``,
    which returns a 3-tuple ``(out, logits, lq_feat)``; only ``out`` is
    used during inference.

    Args:
        dim_embd: transformer token dim. Default 512.
        n_head: number of self-attention heads. Default 8.
        n_layers: number of transformer layers. Default 9.
        codebook_size: VQGAN codebook size. Default 1024.
        latent_size: positional-embedding token count. Default 256 (16x16).
        connect_list: feature scales to fuse with the encoder. Default
            ``['32', '64', '128', '256']`` matches the published model.
        fix_modules: submodules to freeze when training (kept for API
            parity, ignored at inference).
        vqgan_path: optional path to a standalone VQGAN checkpoint used
            to seed the VQGAN half of the model before the CodeFormer
            weights are loaded. Not needed when loading the unified
            ``codeformer.pth`` directly.
    """

    def __init__(
        self,
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=("32", "64", "128", "256"),
        fix_modules=("quantize", "generator"),
        vqgan_path=None,
    ):
        super().__init__(512, 64, (1, 2, 2, 4, 4, 8), "nearest", 2, (16,), codebook_size)

        if vqgan_path is not None:
            self.load_state_dict(torch.load(vqgan_path, map_location="cpu")["params_ema"])

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.connect_list = list(connect_list)
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)
        self.ft_layers = nn.Sequential(
            *[
                TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
                for _ in range(self.n_layers)
            ]
        )
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False),
        )

        self.channels = {
            "16": 512,
            "32": 256,
            "64": 256,
            "128": 128,
            "256": 128,
            "512": 64,
        }
        self.fuse_encoder_block = {"512": 2, "256": 5, "128": 8, "64": 11, "32": 14, "16": 18}
        self.fuse_generator_block = {"16": 6, "32": 9, "64": 12, "128": 15, "256": 18, "512": 21}

        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0.0, detach_16=True, code_only=False, adain=True):
        """Run the CodeFormer inference pass.

        Args:
            x: ``(B, 3, H, W)`` aligned face tensor in ``[-1, 1]``.
            w: fidelity weight in ``[0, 1]``. ``w=0`` = pure codebook
                output (sharpest, least identity-faithful), ``w=1`` =
                keeps encoder features visible. The README suggests
                ``0.5`` for a balanced default. Internally the SFT
                blocks use ``min(w, 1)`` and a no-op branch for
                ``w=0`` so the gradient-free path stays fast.
            detach_16: stop-grad on the codebook features before the SFT
                blocks. Upstream default ``True``.
            code_only: training-only escape hatch; ignored here.
            adain: apply adaptive instance norm so generated features
                match the input's color statistics. Upstream default
                ``True``; turning it off can yield a more "CodeFormer-
                style" face at the cost of color drift.
        """
        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = x
        # ################# Transformer ###################
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        logits = self.idx_pred_layer(query_emb)
        logits = logits.permute(1, 0, 2)

        if code_only:
            return logits, lq_feat

        # ################# Quantization ###################
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
        if detach_16:
            quant_feat = quant_feat.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list:
                f_size = str(x.shape[-1])
                if w > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        return x, logits, lq_feat