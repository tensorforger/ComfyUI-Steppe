"""
Converts STDev development checkpoints (safetensors with `vae` / `architecture` / `model_config` metadata) into
released Steppe model files: release metadata, weights in bfloat16 (inference dtype), release file names.
Also writes the `config.json` catalog of the model repository.

Usage:
    python scripts/export_release.py --latent-space wan_2_1 --source-dir ../STDev/models/latent_upscale_models --output-dir ../Steppe-Wan2.1
    python scripts/export_release.py --latent-space flux_2 --source-dir ../STDev/models/latent_upscale_models --output-dir ../Steppe-Flux.2
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steppe import catalog  # noqa: E402
from steppe.model import make_release_metadata, parse_model_info, read_metadata  # noqa: E402

WAN_2_1_RELEASES = (
    {
        "source_pattern": "wan_2_1_wan_distill_v1_ema_step*.safetensors",
        "file_name": "steppe_wan_2_1_200M_1step.safetensors",
        "title": "Steppe Wan 2.1 200M 1-step",
        "description": "Single step generative 2x latent upscaler for Wan 2.1 VAE latents (images and videos), distilled from the 200M base model.",
        "recommended": {"hff": 0.9, "guidance": 1.0},
    },
    {
        "source_pattern": "wan_2_1_wan_pretrain_v1_ema_step*.safetensors",
        "file_name": "steppe_wan_2_1_200M_base.safetensors",
        "title": "Steppe Wan 2.1 200M base",
        "description": "Multi step (rectified flow) generative 2x latent upscaler for Wan 2.1 VAE latents (images and videos), classifier free guidance on the high frequency feature.",
        "recommended": {"hff": 0.9, "guidance": 1.0, "steps": 20, "shift": 4.0},
    },
    {
        "source_pattern": "wan_2_1_wan_small_s20m_ema_step*.safetensors",
        "file_name": "steppe_wan_2_1_20M_1step.safetensors",
        "title": "Steppe Wan 2.1 20M 1-step",
        "description": "Lite single step generative 2x latent upscaler for Wan 2.1 VAE latents (images and videos).",
        "recommended": {"hff": 0.9, "guidance": 1.0},
    },
)

# FLUX.2: raw weights (better than EMA after the LR decay, see the STDev reports)
FLUX_2_RELEASES = (
    {
        "source_pattern": "flux_2_flux_distill_v1_raw_step*.safetensors",
        "file_name": "steppe_flux_2_200M_1step.safetensors",
        "title": "Steppe FLUX.2 200M 1-step",
        "description": "Single step generative 2x latent upscaler for FLUX.2 VAE latents (images), distilled from the 200M base model.",
        "recommended": {"hff": 0.9, "guidance": 1.0},
    },
    {
        "source_pattern": "flux_2_flux_pretrain_v1_raw_step*.safetensors",
        "file_name": "steppe_flux_2_200M_base.safetensors",
        "title": "Steppe FLUX.2 200M base",
        "description": "Multi step (rectified flow) generative 2x latent upscaler for FLUX.2 VAE latents (images), classifier free guidance on the high frequency feature.",
        "recommended": {"hff": 0.9, "guidance": 1.0, "steps": 30, "shift": 2.0},
    },
    {
        "source_pattern": "flux_2_flux_small_s20m_raw_step*.safetensors",
        "file_name": "steppe_flux_2_20M_1step.safetensors",
        "title": "Steppe FLUX.2 20M 1-step",
        "description": "Lite single step generative 2x latent upscaler for FLUX.2 VAE latents (images).",
        "recommended": {"hff": 0.9, "guidance": 1.0},
    },
)

RELEASES = {"wan_2_1": WAN_2_1_RELEASES, "flux_2": FLUX_2_RELEASES}


def newest(directory: Path, pattern: str) -> Path:
    paths = sorted(glob.glob(str(directory / pattern)))
    if not paths:
        raise FileNotFoundError(f"no {pattern} in {directory}")
    return Path(paths[-1])


def export(source: Path, release: dict, output_directory: Path, version: str) -> dict:
    metadata = read_metadata(str(source))
    info = parse_model_info(str(source), metadata)
    assert info is not None, f"{source} has no development metadata"
    info = replace(info, name=Path(release["file_name"]).stem, version=version)
    tensors = {name: tensor.to(torch.bfloat16).contiguous() for name, tensor in load_file(str(source)).items()}
    release_metadata = make_release_metadata(info, release["title"], release["description"])
    release_metadata["steppe.source"] = f"{metadata.get('run', '?')} step {metadata.get('step', '?')} {metadata.get('weights', '')}".strip()
    output_path = output_directory / release["file_name"]
    save_file(tensors, str(output_path), metadata=release_metadata)
    parameters = sum(tensor.numel() for tensor in tensors.values())
    print(f"{source.name} -> {output_path} ({parameters / 1e6:.1f}M parameters, {os.path.getsize(output_path) / 1e6:.0f} MB)")
    return {
        "version": version,
        "kind": info.kind,
        "latent_space": info.latent_space,
        "parameters": parameters,
        "description": release["description"],
        "recommended_settings": release["recommended"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-space", choices=sorted(RELEASES), default="wan_2_1")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="1.0")
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    releases = RELEASES[arguments.latent_space]
    models = {}
    for release in releases:
        entry = catalog.CATALOG_BY_FILE_NAME[release["file_name"]]
        assert entry.latent_space == arguments.latent_space, release["file_name"]
        source = newest(arguments.source_dir, release["source_pattern"])
        models[release["file_name"]] = export(source, release, arguments.output_dir, arguments.version)
    repository_catalog = {
        "format": "steppe-catalog",
        "format_version": 1,
        "project": "Steppe",
        "latent_space": arguments.latent_space,
        "default_model": releases[0]["file_name"],
        "implementation": "https://github.com/TensorForger/ComfyUI-Steppe",
        "models": models,
    }
    catalog_path = arguments.output_dir / catalog.CATALOG_FILE_NAME
    catalog_path.write_text(json.dumps(repository_catalog, indent=2) + "\n", encoding="utf-8")
    print(f"catalog -> {catalog_path}")


if __name__ == "__main__":
    main()
