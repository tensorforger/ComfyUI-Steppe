"""
Steppe FLUX.2 latent upscaler network (inference only, images): plain window transformer on the full resolution
latent grid (32 channels, 1/8 of the image) with adaLN-zero blocks, shifted 8x8 window attention with relative
position bias and QK-norm, GELU MLP with a depthwise 3x3 convolution, RMS normalized condition vector, learned
source path and residual flow parameterization with a per channel residual table stored in the model file.

The same network serves both model kinds: the multi step flow model (velocity at any sigma, classifier free
guidance on the high frequency feature) and the single step students (extra guidance condition, always sigma 1).
It is spatial only: frames of a [B, C, T, H, W] input are upscaled independently.

Self-contained on purpose (no code shared with the Wan 2.1 network), so that changes for one latent space never
change the results of another.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

LATENT_CHANNELS = 32
WINDOW_SIZE = 8
FREQUENCY_DIM = 256
SIGMA_SCALE = 1000.0
HFF_SCALE = 1000.0
GUIDANCE_SCALE = 500.0
MAX_NORM_GROUPS = 32


@dataclass(frozen=True)
class FluxTransformerConfig:
    width: int = 768
    depth: int = 18
    head_dim: int = 64
    mlp_hidden: int = 3072
    condition_dim: int = 512
    source_encoder_width: int = 512
    source_encoder_blocks: int = 2
    residual_std: tuple[float, ...] = (1.0,) * LATENT_CHANNELS

    def __post_init__(self) -> None:
        object.__setattr__(self, "residual_std", tuple(float(value) for value in self.residual_std))
        assert len(self.residual_std) == LATENT_CHANNELS, len(self.residual_std)


def align_source(latents_source: torch.Tensor) -> torch.Tensor:
    """
    args:
        latents_source: [N, C, h, w]
    returns:
        latents_aligned: [N, C, 2h, 2w], bilinear
    """
    return functional.interpolate(latents_source, scale_factor=2, mode="bilinear", align_corners=False)


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(MAX_NORM_GROUPS, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)


def sinusoidal_embedding(values: torch.Tensor, scale: float) -> torch.Tensor:
    """
    args:
        values: [N]
    returns:
        embedding: [N, FREQUENCY_DIM], dtype: torch.float32
    """
    half_dim = FREQUENCY_DIM // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=values.device, dtype=torch.float32) / half_dim
    )
    angles = values.float()[:, None] * scale * frequencies[None, :]  # [N, half_dim]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def make_projection(condition_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(FREQUENCY_DIM, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim)
    )


class ConditioningEncoder(nn.Module):
    """
    RMS normalized sum of MLP embeddings of the noise level, the high frequency feature (or its learned null
    embedding) and, for single step students, the distilled guidance scale.
    """

    def __init__(self, condition_dim: int, guided: bool) -> None:
        super().__init__()
        self.sigma_projection = make_projection(condition_dim)
        self.hff_projection = make_projection(condition_dim)
        self.null_hff = nn.Parameter(torch.zeros(condition_dim))
        if guided:
            self.guidance_projection = make_projection(condition_dim)

    def forward(
        self,
        sigmas: torch.Tensor,
        hff: torch.Tensor,
        hff_mask: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        args:
            sigmas: [B], values: [0...1], 1 is pure noise
            hff: [B], values: [0...1]
            hff_mask: [B], dtype: torch.bool, False = null hff embedding
            guidance: [B], >= 1, only for guided (single step) models
        returns:
            condition: [B, condition_dim]
        """
        weight_dtype = self.sigma_projection[0].weight.dtype
        sigma_embedding = self.sigma_projection(sinusoidal_embedding(sigmas, SIGMA_SCALE).to(weight_dtype))
        hff_embedding = self.hff_projection(sinusoidal_embedding(hff, HFF_SCALE).to(weight_dtype))
        hff_embedding = torch.where(
            hff_mask[:, None], hff_embedding, self.null_hff.to(hff_embedding.dtype)[None]
        )
        condition = sigma_embedding + hff_embedding
        if hasattr(self, "guidance_projection"):
            assert guidance is not None, "single step models need a guidance value"
            guidance_embedding = self.guidance_projection(
                sinusoidal_embedding(guidance - 1.0, GUIDANCE_SCALE).to(weight_dtype)
            )
            condition = condition + guidance_embedding
        return functional.rms_norm(condition, (condition.shape[-1],))


class PlainResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm_first = make_group_norm(channels)
        self.conv_first = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_second = make_group_norm(channels)
        self.conv_second = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_first(functional.silu(self.norm_first(features)))
        return features + self.conv_second(functional.silu(self.norm_second(hidden)))


class SourceEncoder(nn.Module):
    """
    Learned 2x upsampling of the source latents: residual blocks at the source resolution, pixel shuffle,
    projection to the transformer width.
    """

    def __init__(self, channels: int, output_channels: int, num_blocks: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(LATENT_CHANNELS, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(PlainResidualBlock(channels) for _ in range(num_blocks)))
        self.upsample_norm = make_group_norm(channels)
        self.upsample = nn.Conv2d(channels, 4 * channels, 3, padding=1)
        self.output = nn.Conv2d(channels, output_channels, 1)

    def forward(self, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents_source: [N, C, h, w]
        returns:
            features: [N, output_channels, 2h, 2w]
        """
        features = self.blocks(self.input_conv(latents_source))
        features = functional.pixel_shuffle(self.upsample(functional.silu(self.upsample_norm(features))), 2)
        return self.output(features)


def window_partition(tokens: torch.Tensor) -> torch.Tensor:
    """
    args:
        tokens: [N, H, W, D]
    returns:
        windows: [N, windows, WINDOW_SIZE * WINDOW_SIZE, D]
    """
    count, height, width, dim = tokens.shape
    rows, columns = height // WINDOW_SIZE, width // WINDOW_SIZE
    tokens = tokens.reshape(count, rows, WINDOW_SIZE, columns, WINDOW_SIZE, dim).permute(0, 1, 3, 2, 4, 5)
    return tokens.reshape(count, rows * columns, WINDOW_SIZE * WINDOW_SIZE, dim)


def window_merge(windows: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    args:
        windows: [N, windows, WINDOW_SIZE * WINDOW_SIZE, D]
    returns:
        tokens: [N, H, W, D]
    """
    count, _, _, dim = windows.shape
    rows, columns = height // WINDOW_SIZE, width // WINDOW_SIZE
    windows = windows.reshape(count, rows, columns, WINDOW_SIZE, WINDOW_SIZE, dim).permute(0, 1, 3, 2, 4, 5)
    return windows.reshape(count, height, width, dim)


def shifted_window_mask(height: int, width: int, shift: int, device: torch.device) -> torch.Tensor:
    """
    Swin mask for cyclically shifted windows: tokens attend only within their original region.

    returns:
        mask: [windows, WINDOW_SIZE ** 2, WINDOW_SIZE ** 2], dtype: torch.bool, True = attend
    """
    region = torch.zeros(height, width, device=device)
    label = 0
    for row_slice in (slice(0, -WINDOW_SIZE), slice(-WINDOW_SIZE, -shift), slice(-shift, None)):
        for column_slice in (slice(0, -WINDOW_SIZE), slice(-WINDOW_SIZE, -shift), slice(-shift, None)):
            region[row_slice, column_slice] = label
            label += 1
    region_windows = window_partition(region[None, :, :, None])[0, :, :, 0]  # [windows, window tokens]
    return region_windows[:, :, None] == region_windows[:, None, :]


class TransformerBlock(nn.Module):
    """
    DiT style block with adaLN-zero modulation: (shifted) window self-attention with relative position bias and
    QK-norm, GELU MLP with a depthwise 3x3 convolution on the hidden features.
    """

    def __init__(self, config: FluxTransformerConfig, shift: bool) -> None:
        super().__init__()
        dim = config.width
        self.head_dim = config.head_dim
        self.heads = dim // config.head_dim
        self.shift = WINDOW_SIZE // 2 if shift else 0
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.condition_dim, 6 * dim))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_output = nn.Linear(dim, dim)
        self.query_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        self.key_norm = nn.RMSNorm(config.head_dim, eps=1e-6)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, (2 * WINDOW_SIZE - 1) ** 2))
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(WINDOW_SIZE), torch.arange(WINDOW_SIZE), indexing="ij")
        ).flatten(1)  # [2, window tokens]
        relative = coordinates[:, :, None] - coordinates[:, None, :] + WINDOW_SIZE - 1
        self.register_buffer(
            "relative_index", relative[0] * (2 * WINDOW_SIZE - 1) + relative[1], persistent=False
        )
        self.feed_forward_input = nn.Linear(dim, config.mlp_hidden)
        self.feed_forward_conv = nn.Conv2d(
            config.mlp_hidden, config.mlp_hidden, 3, padding=1, groups=config.mlp_hidden
        )
        self.feed_forward_output = nn.Linear(config.mlp_hidden, dim)

    def _attention(self, tokens: torch.Tensor, shift_mask: torch.Tensor) -> torch.Tensor:
        """
        args:
            tokens: [N, H, W, D]
            shift_mask: [windows, t, t], dtype: torch.bool, used by shifted blocks
        returns:
            attended: [N, H, W, D]
        """
        count, height, width, dim = tokens.shape
        if self.shift:
            tokens = torch.roll(tokens, (-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(self.qkv(tokens))  # [N, windows, t, 3D]
        num_windows, window_tokens = windows.shape[1], windows.shape[2]
        qkv = windows.reshape(count * num_windows, window_tokens, 3, self.heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        qkv = torch.stack([self.query_norm(qkv[0]), self.key_norm(qkv[1]), qkv[2]])
        bias = self.relative_bias[:, self.relative_index].to(qkv.dtype)[None]  # [1, heads, t, t]
        if self.shift:
            bias = bias.masked_fill(~shift_mask[:, None], float("-inf"))  # [windows, heads, t, t]
            bias = bias.expand(count, -1, -1, -1, -1).reshape(
                count * num_windows, self.heads, window_tokens, window_tokens
            )
        attended = functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=bias)
        attended = attended.permute(0, 2, 1, 3).reshape(count, num_windows, window_tokens, dim)
        attended = window_merge(self.attention_output(attended), height, width)
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), dims=(1, 2))
        return attended

    def _feed_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.feed_forward_input(tokens).permute(0, 3, 1, 2)  # [N, F, H, W]
        hidden = self.feed_forward_conv(hidden).permute(0, 2, 3, 1)
        return self.feed_forward_output(functional.gelu(hidden, approximate="tanh"))

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor, shift_mask: torch.Tensor) -> torch.Tensor:
        """
        args:
            tokens: [N, H, W, D]
            condition: [N, condition_dim]
            shift_mask: [windows, t, t], dtype: torch.bool
        returns:
            tokens: [N, H, W, D]
        """
        modulation = self.modulation(condition)[:, None, None, :].chunk(6, dim=-1)  # 6 x [N, 1, 1, D]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation
        tokens = tokens + gate_a * self._attention(self.norm(tokens) * (1 + scale_a) + shift_a, shift_mask)
        return tokens + gate_f * self._feed_forward(self.norm(tokens) * (1 + scale_f) + shift_f)


class SteppeFluxTransformer(nn.Module):
    """
    Velocity model of the residual flow on unpacked, batch norm normalized FLUX.2 latents (32 channels).
    `guided` adds the distilled guidance condition of the single step students.
    """

    supports_temporal = False

    def __init__(self, config: FluxTransformerConfig, guided: bool) -> None:
        super().__init__()
        dim = config.width
        self.config = config
        self.guided = guided
        self.conditioning = ConditioningEncoder(config.condition_dim, guided)
        self.input_conv = nn.Conv2d(2 * LATENT_CHANNELS, dim, 3, padding=1)
        self.source_encoder = SourceEncoder(config.source_encoder_width, dim, config.source_encoder_blocks)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, shift=index % 2 == 1) for index in range(config.depth)
        )
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.condition_dim, 2 * dim))
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.output = nn.Linear(dim, LATENT_CHANNELS)

    @property
    def dtype(self) -> torch.dtype:
        return self.input_conv.weight.dtype

    def from_residual(self, residual: torch.Tensor, latents_source: torch.Tensor) -> torch.Tensor:
        """
        args:
            residual: [B, C, T, 2h, 2w]
            latents_source: [B, C, T, h, w], normalized
        returns:
            latents_target: [B, C, T, 2h, 2w], normalized, dtype: torch.float32
        """
        batch_size, channels, num_frames, height, width = latents_source.shape
        folded = latents_source.float().permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        base = align_source(folded).reshape(batch_size, num_frames, channels, 2 * height, 2 * width)
        scale = torch.tensor(self.config.residual_std, device=residual.device, dtype=torch.float32)
        return base.permute(0, 2, 1, 3, 4) + scale[None, :, None, None, None] * residual.float()

    def forward(
        self,
        noisy_residual: torch.Tensor,
        sigmas: torch.Tensor,
        latents_source: torch.Tensor,
        hff: torch.Tensor,
        hff_mask: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        args:
            noisy_residual: [B, C, T, H, W], H and W multiples of WINDOW_SIZE
            sigmas: [B], values: [0...1], 1 is pure noise
            latents_source: [B, C, T, H / 2, W / 2], normalized FLUX.2 latents (spatial upscaling only)
            hff: [B], values: [0...1]
            hff_mask: [B], dtype: torch.bool, False = null hff embedding
            guidance: [B], >= 1, only for guided (single step) models
        returns:
            velocity: [B, C, T, H, W] (noise - residual), dtype: model dtype
        """
        batch_size, channels, num_frames, height, width = noisy_residual.shape
        assert latents_source.shape[2] == num_frames, "FLUX.2 models upscale spatially only"
        assert (2 * latents_source.shape[3], 2 * latents_source.shape[4]) == (height, width)
        condition = self.conditioning(sigmas, hff, hff_mask, guidance).repeat_interleave(num_frames, dim=0)
        noisy = noisy_residual.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        source = latents_source.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height // 2, width // 2)
        source = source.to(noisy.dtype)
        stacked = torch.cat([noisy, align_source(source)], dim=1)
        features = self.input_conv(stacked) + self.source_encoder(source)  # [N, D, H, W]
        tokens = features.permute(0, 2, 3, 1).contiguous()  # [N, H, W, D]
        del features, stacked
        shift_mask = shifted_window_mask(height, width, WINDOW_SIZE // 2, tokens.device)
        for block in self.blocks:
            tokens = block(tokens, condition, shift_mask)
        shift, scale = self.output_modulation(condition)[:, None, None, :].chunk(2, dim=-1)
        velocity = self.output(self.output_norm(tokens) * (1 + scale) + shift)  # [N, H, W, C]
        return velocity.reshape(batch_size, num_frames, height, width, channels).permute(0, 4, 1, 2, 3)
