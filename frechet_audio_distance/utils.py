from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np


def _process_file(file: Path):
    embd = np.load(file)
    n = embd.shape[0]
    return np.mean(embd, axis=0), np.cov(embd, rowvar=False) * (n - 1), n


def calculate_embd_statistics_online(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings in an online manner.

    :param files: A list of npy files containing ndarrays with shape (n_frames, n_features)
    """
    # Load the first file to get the embedding dimension
    embd_dim = np.load(files[0]).shape[-1]

    # Initialize the mean and covariance matrix
    mu = np.zeros(embd_dim)
    S = np.zeros((embd_dim, embd_dim))  # Sum of squares for online covariance computation
    n = 0  # Counter for total number of frames

    with ProcessPoolExecutor() as executor:
        results = executor.map(_process_file, files)
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