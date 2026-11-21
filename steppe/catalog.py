"""
Released Steppe models and their Hugging Face repositories. Every model repository also has a `config.json`
catalog (model list, versions, recommended settings); it is fetched when a model is loaded to report newer
releases.
"""

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("steppe")

CATALOG_FILE_NAME = "config.json"


@dataclass(frozen=True)
class CatalogEntry:
    file_name: str
    repository: str
    latent_space: str
    kind: str
    version: str
    description: str


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        file_name="steppe_wan_2_1_200M_1step.safetensors",
        repository="TensorForger/Steppe-Wan2.1",
        latent_space="wan_2_1",
        kind="single_step",
        version="1.0",
        description="Default: single step, quality of the 20 step base model.",
    ),
    CatalogEntry(
        file_name="steppe_wan_2_1_200M_base.safetensors",
        repository="TensorForger/Steppe-Wan2.1",
        latent_space="wan_2_1",
        kind="flow",
        version="1.0",
        description="Multi step base model: most control (steps, CFG on hff), research; same quality as the 1-step model.",
    ),
    CatalogEntry(
        file_name="steppe_wan_2_1_20M_1step.safetensors",
        repository="TensorForger/Steppe-Wan2.1",
        latent_space="wan_2_1",
        kind="single_step",
        version="1.0",
        description="Lite: single step, 10x smaller; very large outputs, real time use.",
    ),
)
CATALOG_BY_FILE_NAME = {entry.file_name: entry for entry in CATALOG}


def download_model(entry: CatalogEntry, directory: str) -> str:
    """
    returns:
        local path of the model file, downloaded into `directory` if missing
    """
    from huggingface_hub import hf_hub_download

    logger.info("Steppe: downloading %s from %s", entry.file_name, entry.repository)
    return hf_hub_download(repo_id=entry.repository, filename=entry.file_name, local_dir=directory)


def fetch_repository_catalog(repository: str) -> dict | None:
    """
    returns:
        parsed `config.json` of the model repository (cached by huggingface_hub), None when offline or failing
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repository, filename=CATALOG_FILE_NAME, etag_timeout=5)
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        logger.debug("Steppe: catalog of %s unavailable: %s", repository, error)
        return None


def report_updates(repository: str, file_name: str, local_version: str | None) -> None:
    """
    Logs a notice when the repository catalog lists a newer version of a loaded model.
    """
    catalog = fetch_repository_catalog(repository)
    if catalog is None:
        return
    remote = catalog.get("models", {}).get(file_name)
    if remote is None or local_version is None:
        return
    if parse_version(remote.get("version", "0")) > parse_version(local_version):
        logger.warning(
            "Steppe: %s %s is available (local %s), delete the file to download the new version",
            file_name,
            remote["version"],
            local_version,
        )


def parse_version(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return (0,)
