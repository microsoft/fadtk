"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid

VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import multiprocessing
import os
import random
import subprocess
import tempfile
from typing import Callable
import numpy as np
import torch
from torch import nn
from scipy import linalg
import soundfile as sf
from pathlib import Path
from hypy_utils import write
from hypy_utils.tqdm_utils import tq, pmap, tmap, smap
from hypy_utils.nlp_utils import substr_between

from .models.pann import Cnn14_16k
from .model_loader import ModelLoader


def find_sox_formats(sox_path: str) -> list[str]:
    """
    Find a list of file formats supported by SoX
    """
    out = subprocess.check_output((sox_path, "-h")).decode()
    return substr_between(out, "AUDIO FILE FORMATS: ", "\n").split()


def _cache_embedding_batch(args):
    fs: list[Path]
    ml: ModelLoader
    fs, ml, kwargs = args
    fad = FrechetAudioDistance(ml, **kwargs)
    for f in fs:
        print(f"Loading {f} using {ml.name}")
        fad.cache_embedding_file(f)


def cache_embeddings_files(dir: str | Path, ml_fn: Callable[[], ModelLoader], workers: int = 8, **kwargs):
    """
    Get embeddings for all audio files in a directory.

    :param ml_fn: A function that returns a ModelLoader instance.
    """
    dir = Path(dir)

    # List valid audio files
    files = [dir / f for f in os.listdir(dir)]
    files = [f for f in files if f.is_file()]

    # Randomize order
    random.shuffle(files)

    print(f"[Frechet Audio Distance] Loading {len(files)} audio files from {dir}...")

    # Split files into batches
    batches = list(np.array_split(files, workers))
    
    # Cache embeddings in parallel
    multiprocessing.set_start_method('spawn', force=True)
    with torch.multiprocessing.Pool(workers) as pool:
        pool.map(_cache_embedding_batch, [(b, ml_fn(), kwargs) for b in batches])


class FrechetAudioDistance:
    def __init__(self, ml: ModelLoader, verbose=True, audio_load_worker=8, sox_path="sox"):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ml = ml
        self.ml.load_model()
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
        self.sox_path = sox_path
        self.sox_formats = find_sox_formats(sox_path)

        # Disable gradient calculation because we're not training
        torch.autograd.set_grad_enabled(False)

    def load_audio(self, f: str | Path):
        f = Path(f)

        # Create a directory for storing normalized audio files
        cache_dir = f.parent / "convert" / str(self.ml.sr)
        new = (cache_dir / f.name).with_suffix(".wav")

        if not new.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

            # ffmpeg has bad resampling compared to SoX

            # subprocess.run(["/usr/bin/ffmpeg", 
            #                 "-hide_banner", "-loglevel", "error", 
            #                 "-i", f,
            #                 "-ar", str(self.ml.sr), "-ac", "1", '-acodec', 'pcm_s16le',
            #                 new])
            
            subprocess.run([self.sox_path, f,
                "-r", str(self.ml.sr),
                "-c", "1",
                "-b", "16",
                new])

        if self.ml.name == "encodec":
            import torchaudio
            from encodec.utils import convert_audio

            wav, sr = torchaudio.load(new)
            wav = convert_audio(wav, sr, self.ml.sr, self.ml.model.channels)

            # If it's longer than 3 minutes, cut it
            if wav.shape[1] > 3 * 60 * self.ml.sr:
                wav = wav[:, :3 * 60 * self.ml.sr]
                
            return wav.unsqueeze(0)

        wav_data, _ = sf.read(new, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        return wav_data

    def cache_embedding_file(self, audio_dir: str | Path) -> np.ndarray:
        """
        Compute embedding for an audio file and cache it to a file.
        """
        audio_dir = Path(audio_dir)
        cache = audio_dir.parent / "embeddings" / self.ml.name / audio_dir.with_suffix(".npy").name

        if cache.exists():
            return np.load(cache)

        # Load file
        wav_data = self.load_audio(audio_dir)
        
        # Compute embedding
        embd = self.ml.get_embedding(wav_data)
        
        # Save embedding
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)

        return embd
    
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
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
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    
    def get_embeddings_files(self, dir: str | Path):
        """
        Get embeddings for all audio files in a directory.
        """
        dir = Path(dir)

        # List valid audio files
        files = [dir / f for f in os.listdir(dir)]
        files = [f for f in files if f.is_file()]

        if self.verbose:
            print(f"[Frechet Audio Distance] Loading {len(files)} audio files from {dir}...")

        # Map load task
        # multiprocessing.set_start_method('spawn', force=True)
        # with torch.multiprocessing.Pool(self.audio_load_worker) as pool:
        #     embd_lst = pool.map(self.cache_embedding_file, files)
        embd_lst = tmap(self.cache_embedding_file, files, disable=(not self.verbose), desc="Loading audio files...", max_workers=self.audio_load_worker)
        # embd_lst = smap(self.cache_embedding_file, files)

        return np.concatenate(embd_lst, axis=0)

    def score(self, background_dir, eval_dir):
        try:
            embds_background = self.get_embeddings_files(background_dir)
            embds_eval = self.get_embeddings_files(eval_dir)

            if len(embds_background) == 0 or len(embds_eval) == 0:
                print("[Frechet Audio Distance] background or eval set is empty, exitting...")
                return -1
            
            mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background, 
                sigma_background, 
                mu_eval, 
                sigma_eval
            )

            return fad_score
            
        except Exception as e:
            print("[Frechet Audio Distance] exception thrown, {}".format(str(e)))
            raise e