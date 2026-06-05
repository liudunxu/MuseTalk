"""Vendored subset of CodeFormer's VQGAN architecture.

This file is a minimal, self-contained vendoring of the VQGAN blocks that
CodeFormer needs at inference time. The original lives in
``basicsr/archs/vqgan_arch.py`` of the upstream
https://github.com/sczhou/CodeFormer project (NTU S-Lab License 1.0).

We vendor only the layers that are actually traversed by the published
``codeformer.pth`` checkpoint:
  * ``normalize`` (GroupNorm helper)
  * ``swish``
  * ``VectorQuantizer`` (with ``get_codebook_feat`` used by CodeFormer.forward)
  * ``Downsample`` / ``Upsample`` / ``ResBlock`` / ``AttnBlock``
  * ``Encoder`` / ``Generator`` / ``VQAutoEncoder``

The discriminator (``VQGANDiscriminator``) is intentionally omitted -- we
never instantiate it for inference. ``ARCH_REGISTRY`` and the
``get_root_logger`` import are also removed; this module is plain PyTorch.

Why vendor instead of installing ``basicsr``?
  * ``basicsr`` pulls in ``facexlib`` / ``lmdb`` / ``tb-nightly`` / ``yapf``
    transitively, which conflicts with the leaner runtime.
  * The upstream install also requires ``python basicsr/setup.py develop``,
    which is fragile across Python versions.
  * Vendoring keeps the integration zero-dep beyond ``torch`` / ``numpy``,
    which is what the project already has.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(in_channels: int) -> nn.GroupNorm:
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


@torch.jit.script
def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, emb_dim: int, beta: float):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z: torch.Tensor):
        # z: (B, C, H, W) -> (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)
        d = (
            (z_flattened ** 2).sum(dim=1, keepdim=True)
            + (self.embedding.weight ** 2).sum(1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        mean_distance = torch.mean(d)
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance,
        }

    def get_codebook_feat(self, indices: torch.Tensor, shape):
        # indices: (B*T, 1) flat; shape: (B, H, W, C) target
        indices = indices.view(-1, 1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size, device=indices.device)
        min_encodings.scatter_(1, indices, 1)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        return z_q


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = swish(self.norm1(x_in))
        x = self.conv1(x)
        x = swish(self.norm2(x))
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** -0.5)
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = [nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)]
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)

        blocks = [nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1)]
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VQAutoEncoder(nn.Module):
    """VQGAN encoder+generator pair. The CodeFormer checkpoint embeds the
    VQGAN weights under a ``quantize`` / ``generator`` / ``encoder`` prefix
    matching this layout, so a single ``load_state_dict`` of the full
    CodeFormer checkpoint populates everything."""

    def __init__(
        self,
        img_size,
        nf,
        ch_mult,
        quantizer="nearest",
        res_blocks=2,
        attn_resolutions=(16,),
        codebook_size=1024,
        emb_dim=256,
        beta=0.25,
    ):
        super().__init__()
        self.in_channels = 3
        self.nf = nf
        self.n_blocks = res_blocks
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = list(attn_resolutions)
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
        )
        if self.quantizer_type == "nearest":
            self.beta = beta
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        else:
            raise ValueError(
                f"Unsupported quantizer {quantizer!r}; CodeFormer inference uses 'nearest'"
            )
        self.generator = Generator(
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats