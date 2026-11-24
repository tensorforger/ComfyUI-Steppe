"""
Parity of the production implementation with the STDev research code: the same checkpoint, latents, settings and
seed must give the same upscaled latents. Needs a CUDA GPU, the STDev repository (STDEV_PATH, default
../STDev) and its development checkpoints in STDEV_PATH/models/latent_upscale_models.

Usage: python tests/test_parity.py
"""

import glob
import os
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STDEV_PATH = Path(os.environ.get("STDEV_PATH", REPOSITORY_ROOT.parent / "STDev"))
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(STDEV_PATH / "src"))

from steppe import UpscaleSettings, load_upscaler_file, wan_transformer  # noqa: E402
from steppe.latent_spaces import FLUX_2  # noqa: E402
from stdev.comfy_nodes.flux_2 import sampling as flux_sampling  # noqa: E402
from stdev.comfy_nodes.wan_2_1 import distilled_sampling, sampling  # noqa: E402
from stdev.flux_distill.student import FluxStudentTransformer, upscale_single_step  # noqa: E402
from stdev.flux_pretrain.model import FluxUpscalerTransformer  # noqa: E402
from stdev.flux_pretrain.model import ModelConfig as FluxModelConfig  # noqa: E402
from stdev.wan_distill.student import WanStudentTransformer  # noqa: E402
from stdev.wan_pretrain.model import ModelConfig, WanUpscalerTransformer  # noqa: E402

CHECKPOINT_DIRECTORY = STDEV_PATH / "models" / "latent_upscale_models"
TOLERANCE = 0.02
WINDOW_TOLERANCE = 0.1
CASES = (
    ("image, padded grid", (1, 16, 1, 30, 44), True, False),
    ("video, spatial and temporal", (1, 16, 5, 24, 32), True, True),
    ("video, temporal only", (2, 16, 4, 32, 40), False, True),
)

# ComfyUI FLUX.2 latents: [B, 128, H / 16, W / 16], 2x2 packed and batch norm normalized
FLUX_CASES = (
    ("image, padded grid", (1, 128, 15, 22)),
    ("batch of 2", (2, 128, 16, 16)),
)


def newest_checkpoint(pattern: str) -> str:
    paths = sorted(glob.glob(str(CHECKPOINT_DIRECTORY / pattern)))
    assert paths, f"no checkpoint matching {pattern} in {CHECKPOINT_DIRECTORY}"
    return paths[-1]


def load_reference(path: str, student: bool) -> torch.nn.Module:
    upscaler = load_upscaler_file(path, torch.float32)
    architecture = WanStudentTransformer if student else WanUpscalerTransformer
    reference = architecture(ModelConfig(**upscaler.info.config))
    reference.load_state_dict(upscaler.model.state_dict())
    return reference.to(torch.bfloat16).eval().requires_grad_(False)


def raw_latents(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    mean = torch.tensor(sampling.LATENTS_MEAN)[None, :, None, None, None]
    std = torch.tensor(sampling.LATENTS_STD)[None, :, None, None, None]
    return (torch.randn(shape, generator=generator) * std + mean).to(device)


def report(name: str, ours: torch.Tensor, reference: torch.Tensor, tolerance: float = TOLERANCE) -> bool:
    assert ours.shape == reference.shape, (name, ours.shape, reference.shape)
    difference = (ours - reference).abs()
    relative = (difference.mean() / reference.abs().mean()).item()
    passed = relative < tolerance
    print(f"  {'ok  ' if passed else 'FAIL'} {name}: shape {tuple(ours.shape)}, mean relative difference {relative:.5f}, max {difference.max().item():.4f}")
    return passed


def check_flow_model(device: torch.device) -> bool:
    path = newest_checkpoint("wan_2_1_wan_pretrain_v1_*.safetensors")
    print(f"flow model {Path(path).name}")
    ours = load_upscaler_file(path, torch.bfloat16)
    ours.model.to(device)
    reference = load_reference(path, student=False).to(device)
    passed = True
    for name, shape, spatial, temporal in CASES:
        latents = raw_latents(shape, device)
        noise_scale = 1.0 if spatial else 0.8
        reference_settings = sampling.UpscaleSettings(
            spatial=spatial, temporal=temporal, steps=4, cfg=3.0, hff=0.9, negative_hff=None,
            shift=4.0, noise_scale=noise_scale, seed=7,
        )
        expected = sampling.upscale_latents(reference, latents, reference_settings, device, lambda: None)
        settings = UpscaleSettings(spatial=spatial, temporal=temporal, steps=4, guidance=3.0, hff=0.9, seed=7)
        passed &= report(name, ours.upscale(latents, settings), expected)
    return passed


def check_student(pattern: str, device: torch.device) -> bool:
    path = newest_checkpoint(pattern)
    print(f"single step model {Path(path).name}")
    ours = load_upscaler_file(path, torch.bfloat16)
    ours.model.to(device)
    reference = load_reference(path, student=True).to(device)
    passed = True
    for name, shape, spatial, temporal in CASES:
        for hff in (0.9, None):
            latents = raw_latents(shape, device)
            reference_settings = distilled_sampling.DistilledSettings(
                spatial=spatial, temporal=temporal, guidance=2.0, hff=hff, seed=3
            )
            expected = distilled_sampling.upscale_latents_single_step(
                reference, latents, reference_settings, device, lambda: None
            )
            settings = UpscaleSettings(spatial=spatial, temporal=temporal, guidance=2.0, hff=hff, seed=3)
            passed &= report(f"{name}, hff {hff}", ours.upscale(latents, settings), expected)
    return passed


def check_temporal_windows(device: torch.device) -> bool:
    path = newest_checkpoint("wan_2_1_wan_small_s20m_*.safetensors")
    print("temporal windows (an approximation: random latents, loose tolerance)")
    ours = load_upscaler_file(path, torch.bfloat16)
    ours.model.to(device)
    latents = raw_latents((1, 16, 11, 16, 16), device)
    passed = True
    for spatial, temporal in ((True, True), (True, False), (False, True)):
        whole = ours.upscale(latents, UpscaleSettings(spatial=spatial, temporal=temporal, seed=1))
        windowed = ours.upscale(
            latents,
            UpscaleSettings(spatial=spatial, temporal=temporal, seed=1, temporal_window=4, temporal_context=2),
        )
        passed &= report(f"spatial {spatial}, temporal {temporal}", windowed, whole, WINDOW_TOLERANCE)
    return passed


def check_chunking(device: torch.device) -> bool:
    path = newest_checkpoint("wan_2_1_wan_distill_v1_*.safetensors")
    print("frame and row chunking (must be exact)")
    ours = load_upscaler_file(path, torch.bfloat16)
    ours.model.to(device)
    latents = raw_latents((1, 16, 5, 24, 32), device)
    settings = UpscaleSettings(spatial=True, temporal=True, seed=5)
    whole = ours.upscale(latents, settings)
    original = wan_transformer.TOKENS_PER_CHUNK
    wan_transformer.TOKENS_PER_CHUNK = 1000
    try:
        chunked = ours.upscale(latents, settings)
    finally:
        wan_transformer.TOKENS_PER_CHUNK = original
    return report("chunked vs whole", chunked, whole, 1e-3)


def load_flux_reference(path: str, student: bool) -> torch.nn.Module:
    upscaler = load_upscaler_file(path, torch.float32)
    config = dict(upscaler.info.config)
    config["residual_std"] = tuple(config["residual_std"])
    reference = (FluxStudentTransformer if student else FluxUpscalerTransformer)(FluxModelConfig(**config))
    reference.load_state_dict(upscaler.model.state_dict())
    return reference.to(torch.bfloat16).eval().requires_grad_(False)


def flux_latents(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(4321)
    return torch.randn(shape, generator=generator).to(device)


def check_flux_packing() -> bool:
    print("FLUX.2 packing (must be exact)")
    latents = flux_latents((2, 128, 6, 10), torch.device("cpu"))
    unpacked = FLUX_2.normalize(latents[:, :, None])[:, :, 0]
    expected = flux_sampling.unpatchify_latents(latents)
    passed = report("unpack vs STDev unpatchify", unpacked, expected, 1e-6)
    return passed & report("pack(unpack)", FLUX_2.denormalize(unpacked[:, :, None])[:, :, 0], latents, 1e-6)


def check_flux_flow_model(device: torch.device) -> bool:
    path = newest_checkpoint("flux_2_flux_pretrain_v1_raw_*.safetensors")
    print(f"FLUX.2 flow model {Path(path).name}")
    ours = load_upscaler_file(path, torch.bfloat16)
    ours.model.to(device)
    reference = load_flux_reference(path, student=False).to(device)
    passed = True
    for name, shape in FLUX_CASES:
        for guidance in (1.0, 3.0):
            latents = flux_latents(shape, device)
            reference_settings = flux_sampling.UpscaleSettings(
                steps=4, cfg=guidance, hff=0.9, negative_hff=None, shift=2.0, noise_scale=1.0, seed=7
            )
            expected = flux_sampling.upscale_latents(reference, latents, reference_settings, device, lambda: None)
            settings = UpscaleSettings(steps=4, guidance=guidance, hff=0.9, seed=7)
            passed &= report(f"{name}, guidance {guidance}", ours.upscale(latents, settings), expected)
    latents = flux_latents(FLUX_CASES[0][1], device)
    image = ours.upscale(latents, UpscaleSettings(steps=2, seed=1))
    video_layout = ours.upscale(latents[:, :, None], UpscaleSettings(steps=2, seed=1, temporal=True))[:, :, 0]
    passed &= report("5D input, temporal ignored = 4D input", video_layout, image, 1e-6)
    resolved = ours.resolve_settings(UpscaleSettings())
    passed &= resolved.steps == 30 and resolved.shift == 2.0 and not resolved.temporal
    print(f"  {'ok  ' if passed else 'FAIL'} recommended sampler: steps {resolved.steps}, shift {resolved.shift}")
    return passed


def reference_flux_single_step(
    reference: torch.nn.Module, latents: torch.Tensor, hff: float | None, guidance: float, seed: int
) -> torch.Tensor:
    """
    STDev student sampling of ComfyUI latents: unpatchify, pad, per sample CPU noise (seed + index), patchify.
    """
    results = []
    for index in range(latents.shape[0]):
        source = flux_sampling.unpatchify_latents(latents[index : index + 1].float())
        height, width = source.shape[2:]
        padded = flux_sampling.pad_to_window_grid(source)
        generator = torch.Generator().manual_seed(seed + index)
        noise = torch.randn((1, 32, 2 * padded.shape[2], 2 * padded.shape[3]), generator=generator).to(latents.device)
        hff_tensor = torch.full((1,), float("nan") if hff is None else hff, device=latents.device)
        guidance_tensor = torch.full((1,), guidance, device=latents.device)
        upscaled = upscale_single_step(reference, padded, hff_tensor, guidance_tensor, noise)
        results.append(flux_sampling.patchify_latents(upscaled[:, :, : 2 * height, : 2 * width]).cpu())
    return torch.cat(results)


def check_flux_student(pattern: str, device: torch.device) -> bool:
    path = newest_checkpoint(pattern)
    print(f"FLUX.2 single step model {Path(path).name}")
    ours = load_upscaler_file(path, torch.bfloat16)
    ours.model.to(device)
    reference = load_flux_reference(path, student=True).to(device)
    passed = True
    for name, shape in FLUX_CASES:
        for hff, guidance in ((0.9, 2.0), (None, 1.0)):
            latents = flux_latents(shape, device)
            expected = reference_flux_single_step(reference, latents, hff, guidance, seed=3)
            settings = UpscaleSettings(guidance=guidance, hff=hff, seed=3)
            passed &= report(f"{name}, hff {hff}, guidance {guidance}", ours.upscale(latents, settings), expected)
    return passed


def main() -> None:
    device = torch.device("cuda")
    results = [
        check_flow_model(device),
        check_student("wan_2_1_wan_distill_v1_*.safetensors", device),
        check_student("wan_2_1_wan_small_s20m_*.safetensors", device),
        check_temporal_windows(device),
        check_chunking(device),
        check_flux_packing(),
        check_flux_flow_model(device),
        check_flux_student("flux_2_flux_distill_v1_raw_*.safetensors", device),
        check_flux_student("flux_2_flux_small_s20m_raw_*.safetensors", device),
    ]
    print("PASSED" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
