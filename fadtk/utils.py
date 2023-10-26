from pathlib import Path
import subprocess
import numpy as np
from typing import Union

from hypy_utils.nlp_utils import substr_between
from hypy_utils.tqdm_utils import pmap


PathLike = Union[str, Path]


def _process_file(file: PathLike):
    embd = np.load(file)
    n = embd.shape[0]
    return np.mean(embd, axis=0), np.cov(embd, rowvar=False) * (n - 1), n


def calculate_embd_statistics_online(files: list[PathLike]) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings in an online manner.

    :param files: A list of npy files containing ndarrays with shape (n_frames, n_features)
    """
    assert len(files) > 0, "No files provided"

    # Load the first file to get the embedding dimension
    embd_dim = np.load(files[0]).shape[-1]

    # Initialize the mean and covariance matrix
    mu = np.zeros(embd_dim)
    S = np.zeros((embd_dim, embd_dim))  # Sum of squares for online covariance computation
    n = 0  # Counter for total number of frames

    results = pmap(_process_file, files, desc='Calculating statistics')
    for _mu, _S, _n in results:
        delta = _mu - mu
        mu += _n / (n + _n) * delta
        S += _S + delta[:, None] * delta[None, :] * n * _n / (n + _n)
        n += _n

    if n < 2:
        return mu, np.zeros_like(S)
    else:
        cov = S / (n - 1)  # compute the covariance matrix
        return mu, cov
    

def find_sox_formats(sox_path: str) -> list[str]:
    """
    Find a list of file formats supported by SoX
    """
    try:
        out = subprocess.check_output((sox_path, "-h")).decode()
        return substr_between(out, "AUDIO FILE FORMATS: ", "\n").split()
    except:
        return []


def get_cache_embedding_path(model: str, audio_dir: PathLike) -> Path:
    """
    Get the path to the cached embedding npy file for an audio file.

    :param model: The name of the model
    :param audio_dir: The path to the audio file
    """
    audio_dir = Path(audio_dir)
    return audio_dir.parent / "embeddings" / model / audio_dir.with_suffix(".npy").name
