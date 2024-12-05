import os
import shutil
import subprocess
import tempfile
import traceback
from typing import NamedTuple, Union
import numpy as np
import torch
import torchaudio
from scipy import linalg
from numpy.lib.scimath import sqrt as scisqrt
from pathlib import Path
from hypy_utils import write
from hypy_utils.tqdm_utils import tq, tmap
from hypy_utils.logging_utils import setup_logger

from .model_loader import ModelLoader
from .utils import *

log = setup_logger()
sox_path = os.environ.get('SOX_PATH', 'sox')
ffmpeg_path = os.environ.get('FFMPEG_PATH', 'ffmpeg')
TORCHAUDIO_RESAMPLING = True

if not(TORCHAUDIO_RESAMPLING):
    if not shutil.which(sox_path):
        log.error(f"Could not find SoX executable at {sox_path}, please install SoX and set the SOX_PATH environment variable.")
        exit(3)
    if not shutil.which(ffmpeg_path):
        log.error(f"Could not find ffmpeg executable at {ffmpeg_path}, please install ffmpeg and set the FFMPEG_PATH environment variable.")
        exit(3)


class FADInfResults(NamedTuple):
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]


def calc_embd_statistics(embd_lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings.
    """
    assert embd_lst.shape[0] >= 2, (f"FAD requires at least two embedding window frames, you have {embd_lst.shape}."
        " (This probably means that your audio is too short)")
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
    # NOTE: issues with sqrtm for newer scipy versions
    # using eigenvalue method as workaround
    covmean_sqrtm, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    
    # eigenvalue method
    D, V = linalg.eig(cov1.dot(cov2))
    covmean = (V * scisqrt(D)) @ linalg.inv(V)

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
    tr_covmean_sqrtm = np.trace(covmean_sqrtm)
    if np.iscomplexobj(tr_covmean_sqrtm):
        if np.abs(tr_covmean_sqrtm.imag) < 1e-3:
            tr_covmean_sqrtm = tr_covmean_sqrtm.real

    if not(np.iscomplexobj(tr_covmean_sqrtm)):
        delt = np.abs(tr_covmean - tr_covmean_sqrtm)
        if delt > 1e-3:
            log.warning(f'Detected high error in sqrtm calculation: {delt}')

    return (diff.dot(diff) + np.trace(cov1)
            + np.trace(cov2) - 2 * tr_covmean)


class FrechetAudioDistance:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded = False

    def __init__(self, ml: ModelLoader, audio_load_worker=8, load_model=True):
        self.ml = ml
        self.audio_load_worker = audio_load_worker
        self.sox_formats = find_sox_formats(sox_path)

        if load_model:
            self.ml.load_model()
            self.loaded = True

        # Disable gradient calculation because we're not training
        torch.autograd.set_grad_enabled(False)

    def load_audio(self, f: Union[str, Path]):
        f = Path(f)

        # Create a directory for storing normalized audio files
        cache_dir = f.parent / "convert" / str(self.ml.sr)
        new = (cache_dir / f.name).with_suffix(".wav")

        if not new.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            if TORCHAUDIO_RESAMPLING:
                x, fsorig = torchaudio.load(str(f))
                x = torch.mean(x,0).unsqueeze(0) # convert to mono
                resampler = torchaudio.transforms.Resample(
                    fsorig,
                    self.ml.sr,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
                y = resampler(x)
                torchaudio.save(str(new), y, self.ml.sr, encoding="PCM_S", bits_per_sample=16)
            else:                
                sox_args = ['-r', str(self.ml.sr), '-c', '1', '-b', '16']
    
                # ffmpeg has bad resampling compared to SoX
                # SoX has bad format support compared to ffmpeg
                # If the file format is not supported by SoX, use ffmpeg to convert it to wav
    
                if f.suffix[1:] not in self.sox_formats:
                    # Use ffmpeg for format conversion and then pipe to sox for resampling
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp = Path(tmp) / 'temp.wav'
    
                        # Open ffmpeg process for format conversion
                        subprocess.run([
                            ffmpeg_path, 
                            "-hide_banner", "-loglevel", "error", 
                            "-i", f, tmp])
                        
                        # Open sox process for resampling, taking input from ffmpeg's output
                        subprocess.run([sox_path, tmp, *sox_args, new])
                        
                else:
                    # Use sox for resampling
                    subprocess.run([sox_path, f, *sox_args, new])

        return self.ml.load_wav(new)

    def cache_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)

        if cache.exists():
            return

        # Load file, get embedding, save embedding
        wav_data = self.load_audio(audio_dir)
        embd = self.ml.get_embedding(wav_data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)

    def read_embedding_file(self, audio_dir: Union[str, Path]):
        """
        Read embedding from a cached file.
        """
        cache = get_cache_embedding_path(self.ml.name, audio_dir)
        assert cache.exists(), f"Embedding file {cache} does not exist, please run cache_embedding_file first."
        return np.load(cache)
    
    def load_embeddings(self, dir: Union[str, Path], max_count: int = -1, concat: bool = True):
        """
        Load embeddings for all audio files in a directory.
        """
        files = list(Path(dir).glob("*.*"))
        log.info(f"Loading {len(files)} audio files from {dir}...")

        return self._load_embeddings(files, max_count=max_count, concat=concat)

    def _load_embeddings(self, files: list[Path], max_count: int = -1, concat: bool = True):
        """
        Load embeddings for a list of audio files.
        """
        if len(files) == 0:
            raise ValueError("No files provided")

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
    
    def load_stats(self, path: PathLike):
        """
        Load embedding statistics from a directory.
        """
        if isinstance(path, str):
            # Check if it's a pre-computed statistic file
            bp = Path(__file__).parent / "stats"
            stats = bp / (path.lower() + ".npz")
            print(stats)
            if stats.exists():
                path = stats

        path = Path(path)

        # Check if path is a file
        if path.is_file():
            # Load it as a npz
            log.info(f"Loading embedding statistics from {path}...")
            with np.load(path) as data:
                if f'{self.ml.name}.mu' not in data or f'{self.ml.name}.cov' not in data:
                    raise ValueError(f"FAD statistics file {path} doesn't contain data for model {self.ml.name}")
                return data[f'{self.ml.name}.mu'], data[f'{self.ml.name}.cov']

        cache_dir = path / "stats" / self.ml.name
        emb_dir = path / "embeddings" / self.ml.name
        if cache_dir.exists():
            log.info(f"Embedding statistics is already cached for {path}, loading...")
            mu = np.load(cache_dir / "mu.npy")
            cov = np.load(cache_dir / "cov.npy")
            return mu, cov
        
        if not path.is_dir():
            log.error(f"The dataset you want to use ({path}) is not a directory nor a file.")
            exit(1)

        log.info(f"Loading embedding files from {path}...")
        
        mu, cov = calculate_embd_statistics_online(list(emb_dir.glob("*.npy")))
        log.info("> Embeddings statistics calculated.")

        # Save statistics
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "mu.npy", mu)
        np.save(cache_dir / "cov.npy", cov)
        
        return mu, cov

    def score(self, baseline: PathLike, eval: PathLike):
        """
        Calculate a single FAD score between a background and an eval set.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval: Eval matrix or directory containing eval audio files
        """
        mu_bg, cov_bg = self.load_stats(baseline)
        mu_eval, cov_eval = self.load_stats(eval)

        return calc_frechet_distance(mu_bg, cov_bg, mu_eval, cov_eval)

    def score_inf(self, baseline: PathLike, eval_files: list[Path], steps: int = 25, min_n = 500, raw: bool = False):
        """
        Calculate FAD for different n (number of samples) and compute FAD-inf.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_files: list of eval audio files
        :param steps: number of steps to use
        :param min_n: minimum n to use
        :param raw: return raw results in addition to FAD-inf
        """
        log.info(f"Calculating FAD-inf for {self.ml.name}...")
        # 1. Load background embeddings
        mu_base, cov_base = self.load_stats(baseline)
        # If all of the embedding files end in .npy, we can load them directly
        if all([f.suffix == '.npy' for f in eval_files]):
            embeds = [np.load(f) for f in eval_files]
            embeds = np.concatenate(embeds, axis=0)
        else:
            embeds = self._load_embeddings(eval_files, concat=True)
        
        # Calculate maximum n
        max_n = len(embeds)

        # Generate list of ns to use
        ns = [int(n) for n in np.linspace(min_n, max_n, steps)]
        
        results = []
        for n in tq(ns, desc="Calculating FAD-inf"):
            # Select n feature frames randomly (with replacement)
            indices = np.random.choice(embeds.shape[0], size=n, replace=True)
            embds_eval = embeds[indices]
            
            mu_eval, cov_eval = calc_embd_statistics(embds_eval)
            fad_score = calc_frechet_distance(mu_base, cov_base, mu_eval, cov_eval)

            # Add to results
            results.append([n, fad_score])

        # Compute FAD-inf based on linear regression of 1/n
        ys = np.array(results)
        xs = 1 / np.array(ns)
        slope, intercept = np.polyfit(xs, ys[:, 1], 1)

        # Compute R^2
        r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2)

        # Since intercept is the FAD-inf, we can just return it
        return FADInfResults(score=intercept, slope=slope, r2=r2, points=results)
    
    def score_individual(self, baseline: PathLike, eval_dir: PathLike, csv_name: Union[Path, str]) -> Path:
        """
        Calculate the FAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_dir: Directory containing eval audio files
        :param csv_name: Name of the csv file to write the results to
        :return: Path to the csv file
        """
        csv = Path(csv_name)
        if isinstance(csv_name, str):
            csv = Path('data') / f'fad-individual' / self.ml.name / csv_name
        if csv.exists():
            log.info(f"CSV file {csv} already exists, exiting...")
            return csv

        # 1. Load background embeddings
        mu, cov = self.load_stats(baseline)

        # 2. Define helper function for calculating z score
        def _find_z_helper(f):
            try:
                # Calculate FAD for individual songs
                embd = self.read_embedding_file(f)
                mu_eval, cov_eval = calc_embd_statistics(embd)
                return calc_frechet_distance(mu, cov, mu_eval, cov_eval)

            except Exception as e:
                traceback.print_exc()
                log.error(f"An error occurred calculating individual FAD using model {self.ml.name} on file {f}")
                log.error(e)

        # 3. Calculate z score for each eval file
        _files = list(Path(eval_dir).glob("*.*"))
        scores = tmap(_find_z_helper, _files, desc=f"Calculating scores", max_workers=self.audio_load_worker)

        # 4. Write the sorted z scores to csv
        pairs = list(zip(_files, scores))
        pairs = [p for p in pairs if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        write(csv, "\n".join([",".join([str(x).replace(',', '_') for x in row]) for row in pairs]))

        return csv
