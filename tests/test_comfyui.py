"""
End to end test of the nodes inside a ComfyUI installation (no server): Wan 2.1 VAE encode of an image, Steppe
upscaling with every released model, VAE decode, PNGs in tests/output; plus a long video latent for memory and
speed. Run with the ComfyUI Python environment.

Usage: python tests/test_comfyui.py --comfyui ~/comfyui --models-dir ../Steppe-Wan2.1 --image photo.jpg
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "tests" / "output"


def import_node_pack():
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_Steppe", REPOSITORY_ROOT / "__init__.py", submodule_search_locations=[str(REPOSITORY_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_Steppe"] = module
    spec.loader.exec_module(module)
    return module


def load_image(path: Path, size: int) -> torch.Tensor:
    """
    returns:
        image: [1, H, W, 3], pixel_values: [0...1], center crop resized to size x size
    """
    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)[None]


def save_image(image: torch.Tensor, name: str) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    array = (image[0].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(OUTPUT_DIRECTORY / name)
    print(f"  saved tests/output/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--vae", default="wan_2.1_vae.safetensors")
    parser.add_argument("--size", type=int, default=512)
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.comfyui))
    import comfy.model_management
    import folder_paths
    import nodes as comfy_nodes

    folder_paths.add_model_folder_path("latent_upscale_models", str(arguments.models_dir.resolve()), is_default=True)
    pack = import_node_pack()
    steppe_nodes = sys.modules["ComfyUI_Steppe.nodes"]
    mappings = pack.NODE_CLASS_MAPPINGS

    model_names = mappings["SteppeLoadUpscaler"].INPUT_TYPES()["required"]["model_name"][0]
    print(f"model list: {model_names}")
    (vae,) = comfy_nodes.VAELoader().load_vae(arguments.vae)
    image = load_image(arguments.image, arguments.size)
    save_image(image, "input.png")
    (latent,) = comfy_nodes.VAEEncode().encode(vae, image)
    print(f"encoded latent {tuple(latent['samples'].shape)}")
    (decoded,) = comfy_nodes.VAEDecode().decode(vae, latent)
    bicubic = torch.nn.functional.interpolate(decoded.reshape(1, *decoded.shape[-3:]).movedim(-1, 1), scale_factor=2, mode="bicubic").movedim(1, -1)
    save_image(bicubic, "decode_bicubic.png")
    (measured,) = mappings["SteppeMeasureHFF"]().measure(image)
    print(f"hff of the input image: {measured}")

    upscale = mappings["SteppeUpscaleLatent"]()
    for model_name in [name for name in model_names if name.startswith("steppe_")]:
        (upscaler,) = mappings["SteppeLoadUpscaler"]().load(model_name, "auto")
        single_step = upscaler.upscaler.is_single_step
        for hff, guidance in ((0.9, 2.0), (-1.0, 1.0)) if single_step else ((0.9, 1.0), (0.9, 4.0)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            (upscaled,) = upscale.upscale(upscaler, latent, True, False, hff, guidance, 20, 0)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            (decoded,) = comfy_nodes.VAEDecode().decode(vae, upscaled)
            print(f"{model_name} hff {hff} guidance {guidance}: latent {tuple(upscaled['samples'].shape)}, upscale {elapsed * 1000:.0f} ms")
            save_image(decoded.reshape(1, *decoded.shape[-3:]), f"{Path(model_name).stem}_hff{hff}_g{guidance}.png")

    (upscaler,) = mappings["SteppeLoadUpscaler"]().load("steppe_wan_2_1_200M_1step.safetensors", "auto")
    video = {"samples": torch.randn(1, 16, 21, 60, 104) * 1.5}
    advanced = mappings["SteppeUpscaleLatentAdvanced"]()
    for spatial, temporal, window in ((True, True, 0), (True, True, 9), (False, True, 0)):
        comfy.model_management.soft_empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        (upscaled,) = advanced.upscale(upscaler, video, spatial, temporal, 0.9, 2.0, 20, 0, -1.0, 4.0, -1.0, window, 2)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"video 81 frames 480x832, spatial {spatial} temporal {temporal} window {window}: {tuple(upscaled['samples'].shape)}, {elapsed:.1f} s, peak {peak:.1f} GiB")
    del steppe_nodes


if __name__ == "__main__":
    with torch.inference_mode():
        main()
