"""Download required Twin-2K-500 dataset files."""
from __future__ import annotations

from ..config import Config
from ..data_loading import fetch_all


def fetch_data(cfg: Config) -> None:
    print(f"Fetching required files from {cfg.hf_repo} into {cfg.dataset_cache_dir} ...")
    fetch_all(cfg)
    print("Done.")
