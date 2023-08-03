import multiprocessing
import os
from pathlib import Path
from typing import Callable
import numpy as np

import torch

from .fad import log, FrechetAudioDistance
from .model_loader import ModelLoader


def _cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml, kwargs = args
    fad = FrechetAudioDistance(ml, **kwargs)
    for f in fs:
        log.info(f"Loading {f} using {ml.name}")
        fad.cache_embedding_file(f)


def cache_embedding_files_raw(files: list[Path], ml_fn: Callable[[], ModelLoader], workers: int = 8, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param ml_fn: A function that returns a ModelLoader instance.
    """
    log.info(f"[Frechet Audio Distance] Loading {len(files)} audio files...")

    # Split files into batches
    batches = list(np.array_split(files, workers))
    
    # Cache embeddings in parallel
    multiprocessing.set_start_method('spawn', force=True)
    with torch.multiprocessing.Pool(workers) as pool:
        pool.map(_cache_embedding_batch, [(b, ml_fn(), kwargs) for b in batches])


def cache_embeddings_files(dir: str | Path, ml_fn: Callable[[], ModelLoader], workers: int = 8, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param ml_fn: A function that returns a ModelLoader instance.
    """
    dir = Path(dir)

    # List valid audio files
    files = [dir / f for f in os.listdir(dir)]
    files = [f for f in files if f.is_file()]

    cache_embedding_files_raw(files, ml_fn, workers, **kwargs)