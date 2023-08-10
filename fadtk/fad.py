import os
import random
import subprocess
import tempfile
from typing import Literal
import numpy as np
import torch
from scipy import linalg
from pathlib import Path
from hypy_utils import write
from hypy_utils.tqdm_utils import tq, tmap, pmap
from hypy_utils.nlp_utils import substr_between
from hypy_utils.logging_utils import setup_logger

from .model_loader import ModelLoader
from .utils import *

log = setup_logger()


def calc_embd_statistics(embd_lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings.
    """
    return np.mean(embd_lst, axis=0), np.cov(embd_lst, rowvar=False)


def calc_frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- cov1: The covariance matrix over activations for generated samples.
    -- cov2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)

    assert mu1.shape == mu2.shape, \
        f'Training and test mean vectors have different lengths ({mu1.shape} vs {mu2.shape})'
    assert cov1.shape == cov2.shape, \
        f'Training and test covariances have different dimensions ({cov1.shape} vs {cov2.shape})'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        log.info(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(cov1)
            + np.trace(cov2) - 2 * tr_covmean)


class FrechetAudioDistance:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded = False

    def __init__(self, ml: ModelLoader, audio_load_worker=8, sox_path="sox", load_model=True):
        self.ml = ml
        self.audio_load_worker = audio_load_worker
        self.sox_path = sox_path
        self.sox_formats = find_sox_formats(sox_path)

        if load_model:
            self.ml.load_model()
            self.loaded = True

        # Disable gradient calculation because we're not training
        torch.autograd.set_grad_enabled(False)

    def load_audio(self, f: str | Path):
        f = Path(f)

        # Create a directory for storing normalized audio files
        cache_dir = f.parent / "convert" / str(self.ml.sr)
        new = (cache_dir / f.name).with_suffix(".wav")

        if not new.exists():
            sox_args = ['-r', str(self.ml.sr), '-c', '1', '-b', '16']
            cache_dir.mkdir(parents=True, exist_ok=True)

            # ffmpeg has bad resampling compared to SoX
            # SoX has bad format support compared to ffmpeg
            # If the file format is not supported by SoX, use ffmpeg to convert it to wav

            if f.suffix[1:] not in self.sox_formats:
                # Use ffmpeg for format conversion and then pipe to sox for resampling
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp) / 'temp.wav'

                    # Open ffmpeg process for format conversion
                    subprocess.run([
                        "/usr/bin/ffmpeg", 
                        "-hide_banner", "-loglevel", "error", 
                        "-i", f, tmp])
                    
                    # Open sox process for resampling, taking input from ffmpeg's output
                    subprocess.run([self.sox_path, tmp, *sox_args, new])
                    
            else:
                # Use sox for resampling
                subprocess.run([self.sox_path, f, *sox_args, new])

        return self.ml.load_wav(new)
    
    def get_cache_embedding_path(self, audio_dir: str | Path) -> Path:
        """
        Get the path to the cached embedding npy file for an audio file.
        """
        return Path(audio_dir).parent / "embeddings" / self.ml.name / audio_dir.with_suffix(".npy").name

    def cache_embedding_file(self, audio_dir: str | Path):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        cache = self.get_cache_embedding_path(audio_dir)

        if cache.exists():
            return

        # Load file
        wav_data = self.load_audio(audio_dir)
        
        # Compute embedding
        embd = self.ml.get_embedding(wav_data)
        
        # Save embedding
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)

    def read_embedding_file(self, audio_dir: str | Path):
        """
        Read embedding from a cached file.
        """
        cache = self.get_cache_embedding_path(audio_dir)
        assert cache.exists(), f"Embedding file {cache} does not exist, please run cache_embedding_file first."
        return np.load(cache)
    
    def load_embeddings(self, dir: str | Path, max_count: int = -1, concat: bool = True):
        """
        Load embeddings for all audio files in a directory.
        """
        dir = Path(dir)

        # List valid audio files
        files = [dir / f for f in os.listdir(dir)]
        files = [f for f in files if f.is_file()]
        log.info(f"Loading {len(files)} audio files from {dir}...")

        # Load embeddings
        if max_count == -1:
            embd_lst = tmap(self.read_embedding_file, files, desc="Loading audio files...", max_workers=self.audio_load_worker)
        else:
            total_len = 0
            embd_lst = []
            for f in tq(files, "Loading files"):
                embd_lst.append(self.read_embedding_file(f))
                total_len += embd_lst[-1].shape[0]
                if total_len > max_count:
                    break
        
        # Concatenate embeddings if needed
        if concat:
            return np.concatenate(embd_lst, axis=0)
        else:
            return embd_lst, files
    
    def load_stats(self, dir: str | Path):
        """
        Load embedding statistics from a directory.
        """
        dir = Path(dir)
        cache_dir = dir / "stats" / self.ml.name
        emb_dir = dir / "embeddings" / self.ml.name
        if cache_dir.exists():
            log.info(f"Embedding statistics is already cached for {dir}, loading...")
            mu = np.load(cache_dir / "mu.npy")
            cov = np.load(cache_dir / "cov.npy")
            return mu, cov

        log.info(f"Loading embedding files from {dir}...")
        
        mu, cov = calculate_embd_statistics_online(list(emb_dir.glob("*.npy")))
        log.info("> Embeddings statistics calculated.")

        # Save statistics
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "mu.npy", mu)
        np.save(cache_dir / "cov.npy", cov)
        
        return mu, cov

    def score(self, background_dir: Path | str, eval_dir: Path | str):
        """
        Calculate a single FAD score between a background and an eval set.
        """
        mu_bg, cov_bg = self.load_stats(background_dir)
        mu_eval, cov_eval = self.load_stats(eval_dir)

        return calc_frechet_distance(mu_bg, cov_bg, mu_eval, cov_eval)

    def score_different_n(self, background_dir, eval_dir, csv_name: str, per_n: bool, per_song: bool, steps: int = 25, min_inv = 0.0002, max_idx = -1):
        """
        Calculate FAD for different n (number of samples) from the eval set.

        :param background_dir: directory with background audio files
        :param eval_dir: directory with eval audio files
        :param csv_name: name of the csv file to save the results
        :param per_n: if True, use equally distance n, otherwise use equally distance 1/n
        :param steps: number of steps to use
        :param per_song: if True, n means number of songs, otherwise n means number of feature frames
        :param min_inv: minimum 1/n to use
        :param max_idx: maximum feature frame index of the eval set to use, -1 means use all
        """
        csv = Path('data/fad') / str(min_inv) / self.ml.name / ('n-songs' if per_song else 'n-frames') / ('per-n' if per_n else 'per-ninv') / csv_name
        if csv.exists():
            log.info(f"CSV file {csv} already exists, exitting...")
            return
        
        log.info(f"Calculating FAD for different n for {self.ml.name} on {eval_dir}...")

        # 1. Load background embeddings
        mu_background, cov_background = self.load_stats(background_dir)
        
        eval_dir = Path(eval_dir)
        embd_list, files = self.load_embeddings(eval_dir, max_count=max_idx, concat=False)
        
        # Calculate maximum n
        if per_song:
            max_n = len(embd_list)
        else:
            max_n = sum(len(embed) for embed in embd_list)

        # Generate list of ns to use
        if per_n:
            ns = [int(n) for n in np.linspace(int(1 / min_inv), max_n, steps)]
        else: 
            ns = [int(1 / inv) for inv in np.linspace(1 / max_n, min_inv, steps)]

        results = []
        for n in tq(ns, desc="Calculating FAD for different n"):
            if per_song:
                # Select n songs randomly (with replacement)
                embds_eval = np.concatenate(random.choices(embd_list, k=n), axis=0)
            else:
                # Select n feature frames randomly (with replacement)
                indices = np.random.choice(np.concatenate(embd_list, axis=0).shape[0], size=n, replace=True)
                embds_eval = np.concatenate(embd_list, axis=0)[indices]

            # log.info(f"Selected eval shape {embds_eval.shape}")
            
            mu_eval, cov_eval = calc_embd_statistics(embds_eval)

            fad_score = calc_frechet_distance(mu_background, cov_background, mu_eval, cov_eval)

            # Add to results
            results.append([n, fad_score])

            # Write results to csv
            write(csv, "\n".join([",".join([str(x) for x in row]) for row in results]))


    
    def find_z_songs(self, background_dir, eval_dir, csv_name: str, mode: Literal['z', 'individual', 'delta'] = 'delta'):
        """
        Find songs with the minimum or maximum z scores.
        """
        csv = Path('data') / f'fad-{mode}' / self.ml.name / csv_name
        if csv.exists():
            log.info(f"CSV file {csv} already exists, exitting...")
            return

        # 1. Load background embeddings
        mu, cov = self.load_stats(background_dir)

        sigma = None
        if mode == 'z':
            # Compute the standard deviation for each feature.
            # Assuming the covariance matrix is diagonal, the standard deviations are the square root of the diagonal elements.
            sigma = np.sqrt(np.diagonal(cov))

            # Make sure to add a small constant to the denominator to avoid division by zero.
            sigma = np.where(sigma != 0, sigma, np.finfo(float).eps)

        def _find_z_helper(f):
            # Load embedding
            embd = self.read_embedding_file(f)
            try:

                if mode == 'individual':
                    # Calculate FAD for individual songs
                    mu_eval, cov_eval = calc_embd_statistics(embd)
                    if mu_eval.shape == 1 or cov_eval.shape[0] == 1:
                        log.info(f"{f} is incorrect", embd.shape)
                    return calc_frechet_distance(mu, cov, mu_eval, cov_eval)
                
                elif mode == 'z':
                    # Calculate z-score matrix
                    zs = (embd - mu) / sigma
                    
                    # Calculate mean abs z score
                    return np.mean(np.abs(zs))
                
                else:
                    raise ValueError(f"Invalid mode {mode}")

            except Exception as e:
                log.info(f)
                log.info(self.ml.name)
                return None

        if mode != 'delta':
            # 3. Calculate z score for each eval file
            eval_dir = Path(eval_dir)
            _files = [eval_dir / f for f in os.listdir(eval_dir)]
            _files = [f for f in _files if f.is_file()]

            # args = [(self, f, mu, cov, sigma, mode) for f in _files]
            
            scores = tmap(_find_z_helper, _files, desc=f"Calculating {mode} scores...", max_workers=self.audio_load_worker)
            scores = np.array(scores)
        
        else:
            # Delta mode cannot be parallelized because of memory constraints
            # Loop through each index, make a shallow copy, remove it from the eval set, calculate the FAD score
            embd_list, _files = self.load_embeddings(eval_dir, concat=False)
            embd_list = embd_list[:500]
            _files = _files[:500]
            fad_original = calc_frechet_distance(mu, cov, *calc_embd_statistics(np.concatenate(embd_list, axis=0)))

            scores = []
            for i in tq(range(len(embd_list)), desc="Calculating delta scores"):
                # Shallow copy
                embd_list_copy = embd_list.copy()
                embd_list_copy.pop(i)
                mu_eval, cov_eval = calc_embd_statistics(np.concatenate(embd_list_copy, axis=0))
                scores.append(calc_frechet_distance(mu, cov, mu_eval, cov_eval) - fad_original)

        # Write the sorted z scores to csv
        pairs = list(zip(_files, scores))
        # Filter out nones
        pairs = [p for p in pairs if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        write(csv, "\n".join([",".join([str(x).replace(',', '_') for x in row]) for row in pairs]))
