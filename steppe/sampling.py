"""
Latent upscaling with Steppe models: raw VAE latents are normalized, padded to the window grid, upscaled by the
residual flow (Euler steps with classifier free guidance on the high frequency feature) or in one step by a
distilled student, cropped and denormalized. Long videos can be processed in overlapping temporal windows.
"""

from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn.functional as functional

from .flux_transformer import WINDOW_SIZE as FLUX_WINDOW_SIZE
from .flux_transformer import SteppeFluxTransformer
from .latent_spaces import LatentSpace
from .wan_transformer import WINDOW_SIZE, SteppeWanTransformer

UpscalerNetwork = SteppeWanTransformer | SteppeFluxTransformer
assert FLUX_WINDOW_SIZE == WINDOW_SIZE

NOISE_SCALE_SPATIAL = 1.0
NOISE_SCALE_TEMPORAL = 0.8
STUDENT_SIGMA = 1.0


@dataclass(frozen=True)
class UpscaleSettings:
    """
    hff: target high frequency feature (sharpness, [0...1]); None = null embedding, the model decides itself.
    guidance: classifier free guidance scale for multi step models (positive hff vs negative hff), distilled
        guidance input for single step models; 1 = no guidance.
    negative_hff: hff of the negative guidance branch of multi step models; None = null embedding.
    steps, shift: Euler steps and SD3 sigma shift of multi step models (ignored by single step models); None = the
        recommended values of the model (`SteppeUpscaler.resolve_settings`).
    noise_scale: initial noise temperature; None = 1.0 with spatial upscaling, 0.8 for temporal only.
    temporal_window: new source latent frames per window, 0 = the whole clip at once.
    temporal_context: source latent frames of overlap before every window after the first, >= 1.
    """

    spatial: bool = True
    temporal: bool = False
    hff: float | None = 0.9
    guidance: float = 1.0
    negative_hff: float | None = None
    steps: int | None = None
    shift: float | None = None
    noise_scale: float | None = None
    seed: int = 0
    temporal_window: int = 0
    temporal_context: int = 2


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """
    args:
        sigmas: [...], values: [0...1], 1 is pure noise
    returns:
        shifted_sigmas: [...], values: [0...1], moved towards 1 for shift > 1 (SD3)
    """
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def target_size_of(source_size: tuple[int, int, int], spatial: bool, temporal: bool) -> tuple[int, int, int]:
    frames, height, width = source_size
    if temporal:
        frames = 1 + 2 * (frames - 1)
    if spatial:
        height, width = 2 * height, 2 * width
    return frames, height, width


def pad_to_window_grid(latents: torch.Tensor, spatial: bool) -> torch.Tensor:
    """
    Replicate padding (bottom, right) so that the target latent grid is a multiple of the window size.

    args:
        latents: [B, C, T, H, W]
    returns:
        latents: [B, C, T, H_padded, W_padded]
    """
    multiple = WINDOW_SIZE // 2 if spatial else WINDOW_SIZE
    pad_height = -latents.shape[3] % multiple
    pad_width = -latents.shape[4] % multiple
    if pad_height == 0 and pad_width == 0:
        return latents
    return functional.pad(latents, (0, pad_width, 0, pad_height, 0, 0), mode="replicate")


def count_model_passes(model: UpscalerNetwork, settings: UpscaleSettings) -> int:
    """
    returns:
        network evaluations per sample and temporal window
    """
    if model.guided:
        return 1
    return settings.steps * (2 if settings.guidance != 1.0 else 1)


def count_temporal_windows(source_frames: int, settings: UpscaleSettings) -> int:
    if settings.temporal_window <= 0 or source_frames <= settings.temporal_window:
        return 1
    return 1 + -(-(source_frames - settings.temporal_window) // settings.temporal_window)


def hff_inputs(value: float | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
        hff: [1], values: [0...1], hff_mask: [1], dtype: torch.bool, False = null embedding
    """
    hff = torch.full((1,), 0.0 if value is None else float(value), device=device)
    hff_mask = torch.full((1,), value is not None, dtype=torch.bool, device=device)
    return hff, hff_mask


@torch.no_grad()
def sample_flow(
    model: UpscalerNetwork,
    latents_source: torch.Tensor,
    noise: torch.Tensor,
    settings: UpscaleSettings,
    on_pass: Callable[[], None],
) -> torch.Tensor:
    """
    Euler sampling of the residual flow; guidance: negative + guidance * (positive - negative).

    args:
        latents_source: [1, C, T_lr, H_lr, W_lr], normalized
        noise: [1, C, T_hr, H_hr, W_hr], noise_scale * unit gaussian
    returns:
        residual: [1, C, T_hr, H_hr, W_hr], dtype: torch.float32
    """
    device = latents_source.device
    dtype = model.dtype
    samples = noise.float()
    source = latents_source.to(dtype)
    sigmas = shift_sigmas(torch.linspace(1.0, 0.0, settings.steps + 1, device=device), settings.shift)
    positive_hff, positive_mask = hff_inputs(settings.hff, device)
    negative_hff, negative_mask = hff_inputs(settings.negative_hff, device)
    use_guidance = settings.guidance != 1.0
    for current_sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        sigma_batch = current_sigma.expand(1)
        velocity = model(samples.to(dtype), sigma_batch, source, positive_hff, positive_mask).float()
        on_pass()
        if use_guidance:
            negative_velocity = model(samples.to(dtype), sigma_batch, source, negative_hff, negative_mask).float()
            velocity = negative_velocity + settings.guidance * (velocity - negative_velocity)
            on_pass()
        samples = samples + (next_sigma - current_sigma) * velocity
    return samples


@torch.no_grad()
def sample_single_step(
    model: UpscalerNetwork,
    latents_source: torch.Tensor,
    noise: torch.Tensor,
    settings: UpscaleSettings,
    on_pass: Callable[[], None],
) -> torch.Tensor:
    """
    args:
        latents_source: [1, C, T_lr, H_lr, W_lr], normalized
        noise: [1, C, T_hr, H_hr, W_hr], noise_scale * unit gaussian
    returns:
        residual: [1, C, T_hr, H_hr, W_hr], dtype: torch.float32
    """
    device = latents_source.device
    dtype = model.dtype
    hff, hff_mask = hff_inputs(settings.hff, device)
    guidance = torch.full((1,), float(settings.guidance), device=device)
    sigmas = torch.full((1,), STUDENT_SIGMA, device=device)
    velocity = model(noise.to(dtype), sigmas, latents_source.to(dtype), hff, hff_mask, guidance)
    on_pass()
    return noise.float() - velocity.float()


@torch.no_grad()
def upscale_window(
    model: UpscalerNetwork,
    latents_source: torch.Tensor,
    noise: torch.Tensor,
    settings: UpscaleSettings,
    on_pass: Callable[[], None],
) -> torch.Tensor:
    """
    args:
        latents_source: [1, C, T_lr, H_lr, W_lr], normalized, padded to the window grid
        noise: [1, C, T_hr, H_hr, W_hr], noise_scale * unit gaussian
    returns:
        latents: [1, C, T_hr, H_hr, W_hr], normalized, dtype: torch.float32
    """
    sampler = sample_single_step if model.guided else sample_flow
    residual = sampler(model, latents_source, noise, settings, on_pass)
    latents = model.from_residual(residual, latents_source)
    if not settings.spatial:
        latents[:, :, 0] = latents_source[:, :, 0].float()
    return latents


@torch.no_grad()
def upscale_sample(
    model: UpscalerNetwork,
    latents_source: torch.Tensor,
    noise: torch.Tensor,
    settings: UpscaleSettings,
    on_pass: Callable[[], None],
) -> torch.Tensor:
    """
    Whole clip at once or in temporal windows: window k covers new source frames [s, e) plus `temporal_context`
    frames before them; only target frames of the new source frames are kept (target frame j <- source frame
    ceil(j / 2) for temporal upscaling).

    args:
        latents_source: [1, C, T_lr, H_lr, W_lr], normalized, padded to the window grid
        noise: [1, C, T_hr, H_hr, W_hr], noise_scale * unit gaussian for the whole clip
    returns:
        latents: [1, C, T_hr, H_hr, W_hr], normalized, dtype: torch.float32
    """
    source_frames = latents_source.shape[2]
    if count_temporal_windows(source_frames, settings) == 1:
        return upscale_window(model, latents_source, noise, settings, on_pass)
    window, context = settings.temporal_window, max(1, settings.temporal_context)

    def target_index(source_index: int) -> int:
        return 2 * source_index if settings.temporal else source_index

    pieces = []
    for start in range(0, source_frames, window):
        end = min(source_frames, start + window)
        first = max(0, start - context)
        target_first, target_last = target_index(first), target_index(end - 1)
        upscaled = upscale_window(
            model,
            latents_source[:, :, first:end],
            noise[:, :, target_first : target_last + 1],
            settings,
            on_pass,
        )
        keep_from = 0 if start == 0 else target_index(start - 1) + 1 - target_first
        pieces.append(upscaled[:, :, keep_from:])
    return torch.cat(pieces, dim=2)


@torch.no_grad()
def upscale_latents(
    model: UpscalerNetwork,
    latent_space: LatentSpace,
    latents: torch.Tensor,
    settings: UpscaleSettings,
    device: torch.device,
    on_pass: Callable[[], None] = lambda: None,
) -> torch.Tensor:
    """
    Every sample uses its own noise from a CPU generator seeded with seed + sample index (reproducible across
    devices).

    args:
        latents: [B, C, T, H, W], VAE latents as handed over by ComfyUI / diffusers (`LatentSpace`)
        settings: steps and shift resolved for multi step models (not None)
    returns:
        latents: [B, C, T_hr, H_hr, W_hr], VAE latents as handed over, dtype: torch.float32, on the CPU
    """
    assert latents.ndim == 5 and latents.shape[1] == latent_space.channels, (
        f"expected {latent_space.display_name} latents [B, {latent_space.channels}, T, H, W], "
        f"got {tuple(latents.shape)}"
    )
    assert settings.spatial or settings.temporal, "enable spatial and / or temporal upscaling"
    assert model.supports_temporal or not settings.temporal, "this model upscales spatially only"
    assert model.guided or (settings.steps is not None and settings.shift is not None), "resolve steps and shift"
    if settings.noise_scale is None:
        settings = replace(
            settings, noise_scale=NOISE_SCALE_SPATIAL if settings.spatial else NOISE_SCALE_TEMPORAL
        )
    results = []
    for index in range(latents.shape[0]):
        normalized = latent_space.normalize(latents[index : index + 1].to(device))
        target_size = target_size_of(tuple(normalized.shape[2:]), settings.spatial, settings.temporal)
        padded = pad_to_window_grid(normalized, settings.spatial)
        padded_target = target_size_of(tuple(padded.shape[2:]), settings.spatial, settings.temporal)
        generator = torch.Generator().manual_seed(settings.seed + index)
        noise = torch.randn((1, padded.shape[1], *padded_target), generator=generator)
        noise = settings.noise_scale * noise.to(device)
        upscaled = upscale_sample(model, padded, noise, settings, on_pass)
        upscaled = upscaled[:, :, : target_size[0], : target_size[1], : target_size[2]]
        results.append(latent_space.denormalize(upscaled).cpu())
    return torch.cat(results)
