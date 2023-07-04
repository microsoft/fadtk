from abc import ABC, abstractmethod
import os
from typing import Literal
import numpy as np
import requests
import soundfile

import torch
import librosa
from torch import nn
from pathlib import Path
from hypy_utils.downloader import download_file

from .models.pann import Cnn14_16k


class ModelLoader(ABC):
    def __init__(self, name: str):
        self.model = None
        self.sr = None
        self.name = name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @abstractmethod
    def load_model(self):
        pass

    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        return embd

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray):
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path):
        wav_data, _ = soundfile.read(wav_file, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        # print(wav_data.shape, np.mean(wav_data), np.std(wav_data))

        return wav_data


class VGGishModel(ModelLoader):
    """
    S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
    """
    def __init__(self, use_pca=False, use_activation=False):
        super().__init__("vggish")
        self.use_pca = use_pca
        self.use_activation = use_activation

    def load_model(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        if not self.use_pca:
            self.model.postprocess = False
        if not self.use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.sr = 16000
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        return self.model.forward(audio, self.sr)


class PANNModel(ModelLoader):
    """
    Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020)
    """
    def __init__(self):
        super().__init__("pann")

    def load_model(self):
        model_path = os.path.join(torch.hub.get_dir(), "PANNs-FAD.pth")
        if not(os.path.exists(model_path)):
            torch.hub.download_url_to_file('https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth', dst=model_path, progress=True)
        self.model = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.sr = 16000
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model(torch.tensor(audio).float().unsqueeze(0).to(self.device), None)
            d = out['embedding'].data
            print(d.shape)
            return d


class EncodecBaseModel(ModelLoader):
    def __init__(self, name: str):
        super().__init__(name)
    
    def load_model(self):
        from encodec import EncodecModel
        self.model = EncodecModel.encodec_model_24khz()
        self.sr = 24000
        self.model.set_target_bandwidth(12)
        self.model.to(self.device)

    
    def load_wav(self, wav_file: Path):
        import torchaudio
        from encodec.utils import convert_audio

        wav, sr = torchaudio.load(wav_file)
        wav = convert_audio(wav, sr, self.sr, self.model.channels)

        # print(wav.shape, torch.mean(wav), torch.std(wav))

        # If it's longer than 3 minutes, cut it
        if wav.shape[1] > 3 * 60 * self.sr:
            wav = wav[:, :3 * 60 * self.sr]

        return wav.unsqueeze(0)


class EncodecQuantModel(EncodecBaseModel):
    """
    Encodec model from https://github.com/facebookresearch/encodec

    This version uses the quantized outputs (discrete values of n quantizers).
    """
    def __init__(self):
        super().__init__("encodec")

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            frames = self.model.encode(audio.to(self.device))
            # print(frames[0][0].shape) # [batch_size, n_quantizers, timeframes]
            frames = torch.cat([e[0] for e in frames], dim=-1)
            # print(frames.shape) # [batch_size, n_quantizers, timeframes]
            frames = frames[0]
            # print(frames.shape) # [n_quantizers, timeframes]
            frames = frames.transpose(0, 1)
            # print(frames.shape) # [timeframes, n_quantizers]
            return frames


class EncodecEmbModel(EncodecBaseModel):
    """
    Encodec model from https://github.com/facebookresearch/encodec

    Thiss version uses the embedding outputs (continuous values of 128 features).
    """
    def __init__(self):
        super().__init__("encodec-emb")

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            length = audio.shape[-1]
            duration = length / self.sr
            assert self.model.segment is None or duration <= 1e-5 + self.model.segment

            emb = self.model.encoder(audio.to(self.device)) # [1, 128, timeframes]
            emb = emb[0] # [128, timeframes]
            emb = emb.transpose(0, 1) # [timeframes, 128]
            return emb


class MERTModel(ModelLoader):
    """
    MERT model from https://huggingface.co/m-a-p/MERT-v1-330M
    """
    def __init__(self, size='v1-95M'):
        super().__init__(f"MERT-{size}")

    def load_model(self):
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel
        
        self.model = AutoModel.from_pretrained(f"m-a-p/{self.name}", trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"m-a-p/{self.name}",trust_remote_code=True)
        self.sr = self.processor.sampling_rate
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=False)
            out = out.last_hidden_state
        
        # print(out.shape) # [1, timeframes, 768]
        # print(out[-1].shape) # [timeframes, 768]

        return out[-1]
    

class CLAPLaionModel(ModelLoader):
    """
    CLAP model from https://github.com/LAION-AI/CLAP
    """
    def __init__(self, type: Literal['audio', 'music']):
        super().__init__(f"clap-laion-{type}")
        self.type = type

        if type == 'audio':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt'
        elif type == 'music':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'

        self.model_file = Path(__file__).parent / ".model-checkpoints" / url.split('/')[-1]

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)

    def load_model(self):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt(self.model_file)
        self.sr = 48000
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.reshape(1, -1)

        # The int16-float32 conversion is used for quantization
        audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 10s chunks with 1s hop
        chunk_size = 10 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.get_audio_embedding_from_data(x = chunk, use_tensor=True)
                embeddings.append(emb)

        # Concatenate the embeddings
        emb = torch.cat(embeddings, dim=0)

        print(emb.shape) # [timeframes, 512]
        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

    def load_wav(self, wav_file: Path):
        wav_data, _ = librosa.load(wav_file, sr=self.sr)
        return wav_data

