from abc import ABC, abstractmethod
import logging
import math
from typing import Literal
import numpy as np
import soundfile

import torch
import librosa
from torch import nn
from pathlib import Path
from hypy_utils.downloader import download_file
import torch.nn.functional as F
import importlib.util
import importlib.metadata


log = logging.getLogger(__name__)


class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, num_features: int, sr: int, min_len: int = -1):
        """
        Args:
            name (str): A unique identifier for the model.
            num_features (int): Number of features in the output embedding (dimensionality).
            sr (int): Sample rate of the audio.
            min_len (int, optional): Enforce a minimal length for the audio in seconds. Defaults to -1 (no minimum).
        """
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.min_len = min_len
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray):
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path):
        wav_data, _ = soundfile.read(wav_file, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        
        # Enforce minimum length
        wav_data = self.enforce_min_len(wav_data)

        return wav_data
    
    def enforce_min_len(self, audio: np.ndarray) -> np.ndarray:
        """
        Enforce a minimum length for the audio. If the audio is too short, output a warning and pad it with zeros.
        """
        if self.min_len < 0:
            return audio
        if audio.shape[0] < self.min_len * self.sr:
            log.warning(
                f"Audio is too short for {self.name}.\n"
                f"The model requires a minimum length of {self.min_len}s, audio is {audio.shape[0] / self.sr:.2f}s.\n"
                f"Padding with zeros."
            )
            audio = np.pad(audio, (0, int(np.ceil(self.min_len * self.sr - audio.shape[0]))))
            print()
        return audio


class VGGishModel(ModelLoader):
    """
    S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
    """
    def __init__(self, use_pca=False, use_activation=False):
        super().__init__("vggish", 128, 16000, min_len=1)
        self.use_pca = use_pca
        self.use_activation = use_activation

    def load_model(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        if not self.use_pca:
            self.model.postprocess = False
        if not self.use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        return self.model.forward(audio, self.sr)
        

class EncodecEmbModel(ModelLoader):
    """
    Encodec model from https://github.com/facebookresearch/encodec

    Thiss version uses the embedding outputs (continuous values of 128 features).
    """
    def __init__(self, variant: Literal['48k', '24k'] = '24k'):
        super().__init__('encodec-emb' if variant == '24k' else f"encodec-emb-{variant}", 128,
                         sr=24000 if variant == '24k' else 48000)
        self.variant = variant

    def load_model(self):
        from encodec import EncodecModel
        if self.variant == '48k':
            self.model = EncodecModel.encodec_model_48khz()
            self.model.set_target_bandwidth(24)
        else:
            self.model = EncodecModel.encodec_model_24khz()
            self.model.set_target_bandwidth(12)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        segment_length = self.model.segment_length
        
        # The 24k model doesn't use segmenting
        if segment_length is None:
            return self._get_frame(audio)
        
        # The 48k model uses segmenting
        assert audio.dim() == 3
        _, channels, length = audio.shape
        assert channels > 0 and channels <= 2
        stride = segment_length

        encoded_frames: list[torch.Tensor] = []
        for offset in range(0, length, stride):
            frame = audio[:, :, offset:offset + segment_length]
            encoded_frames.append(self._get_frame(frame))

        # Concatenate
        encoded_frames = torch.cat(encoded_frames, dim=0) # [timeframes, 128]
        return encoded_frames

    def _get_frame(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            length = audio.shape[-1]
            duration = length / self.sr
            assert self.model.segment is None or duration <= 1e-5 + self.model.segment, f"Audio is too long ({duration} > {self.model.segment})"

            emb = self.model.encoder(audio.to(self.device)) # [1, 128, timeframes]
            emb = emb[0] # [128, timeframes]
            emb = emb.transpose(0, 1) # [timeframes, 128]
            return emb
    
    def load_wav(self, wav_file: Path):
        import torchaudio
        from encodec.utils import convert_audio

        wav, sr = torchaudio.load(str(wav_file))
        wav = convert_audio(wav, sr, self.sr, self.model.channels)

        # If it's longer than 3 minutes, cut it
        if wav.shape[1] > 3 * 60 * self.sr:
            wav = wav[:, :3 * 60 * self.sr]

        return wav.unsqueeze(0)
        
    def _decode_frame(self, emb: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            emb = torch.from_numpy(emb).float().to(self.device) # [timeframes, 128]
            emb = emb.transpose(0, 1) # [128, timeframes]
            emb = emb.unsqueeze(0) # [1, 128, timeframes]
            audio = self.model.decoder(emb) # [1, 1, timeframes]
            audio = audio[0, 0] # [timeframes]

            return audio.cpu().numpy()


class DACModel(ModelLoader):
    """
    DAC model from https://github.com/descriptinc/descript-audio-codec

    pip install descript-audio-codec
    """
    def __init__(self):
        super().__init__("dac-44kHz", 1024, 44100)

    def load_model(self):
        from dac.utils import load_model
        self.model = load_model(tag='latest', model_type='44khz')
        self.model.eval()
        self.model.to(self.device)

    def _get_embedding(self, audio) -> np.ndarray:
        from audiotools import AudioSignal
        import time

        audio: AudioSignal

        # Set variables
        win_len = 5.0
        overlap_hop_ratio = 0.5

        # Fix overlap window so that it's divisible by 4 in # of samples
        win_len = ((win_len * self.sr) // 4) * 4
        win_len = win_len / self.sr
        hop_len = win_len * overlap_hop_ratio

        stime = time.time()

        # Sanitize input
        audio.normalize(-16)
        audio.ensure_max_of_audio()

        nb, nac, nt = audio.audio_data.shape
        audio.audio_data = audio.audio_data.reshape(nb * nac, 1, nt)

        pad_length = math.ceil(audio.signal_duration / win_len) * win_len
        audio.zero_pad_to(int(pad_length * self.sr))
        audio = audio.collect_windows(win_len, hop_len)

        print(win_len, hop_len, audio.batch_size, f"(processed in {(time.time() - stime) * 1000:.0f}ms)")
        stime = time.time()

        emb = []
        for i in range(audio.batch_size):
            signal_from_batch = AudioSignal(audio.audio_data[i, ...], self.sr)
            signal_from_batch.to(self.device)
            e1 = self.model.encoder(signal_from_batch.audio_data).cpu() # [1, 1024, timeframes]
            e1 = e1[0] # [1024, timeframes]
            e1 = e1.transpose(0, 1) # [timeframes, 1024]
            emb.append(e1)

        emb = torch.cat(emb, dim=0)
        print(emb.shape, f'(computing finished in {(time.time() - stime) * 1000:.0f}ms)')

        return emb

    def load_wav(self, wav_file: Path):
        from audiotools import AudioSignal
        return AudioSignal(wav_file)


class MERTModel(ModelLoader):
    """
    MERT model from https://huggingface.co/m-a-p/MERT-v1-330M

    Please specify the layer to use (1-12).
    """
    def __init__(self, size='v1-95M', layer=12, limit_minutes=6):
        super().__init__(f"MERT-{size}" + ("" if layer == 12 else f"-{layer}"), 768, 24000)
        self.huggingface_id = f"m-a-p/MERT-{size}"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr
        
    def load_model(self):
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel
        
        self.model = AutoModel.from_pretrained(self.huggingface_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.huggingface_id, trust_remote_code=True)
        # self.sr = self.processor.sampling_rate
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Limit to 9 minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze() # [13 layers, timeframes, 768]
            out = out[self.layer] # [timeframes, 768]

        return out


class CLAPLaionModel(ModelLoader):
    """
    CLAP model from https://github.com/LAION-AI/CLAP
    """
    
    def __init__(self, type: Literal['audio', 'music']):
        super().__init__(f"clap-laion-{type}", 512, 48000)
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
            
        # Patch the model file to remove position_ids (will raise an error otherwise)
        # This key must be removed for CLAP version <= 1.1.5
        # But it must be kept for CLAP version >= 1.1.6
        package_name = "laion_clap"
        from packaging import version
        ver = version.parse(importlib.metadata.version(package_name))
        if ver < version.parse("1.1.6"):
            self.patch_model_430(self.model_file)
        else:
            self.unpatch_model_430(self.model_file)


    def patch_model_430(self, file: Path):
        """
        Patch the model file to remove position_ids (will raise an error otherwise)
        This is a new issue after the transformers 4.30.0 update
        Please refer to https://github.com/LAION-AI/CLAP/issues/127
        """
        # Create a "patched" file when patching is done
        patched = file.parent / f"{file.name}.patched.430"
        if patched.exists():
            return
        
        log.warning("Patching LAION-CLAP's model checkpoints")
        
        # Load the checkpoint from the given path
        ck = torch.load(file, map_location="cpu")

        # Extract the state_dict from the checkpoint
        unwrap = isinstance(ck, dict) and "state_dict" in ck
        sd = ck["state_dict"] if unwrap else ck

        # Delete the specific key from the state_dict
        sd.pop("module.text_branch.embeddings.position_ids", None)

        # Save the modified state_dict back to the checkpoint
        if isinstance(ck, dict) and "state_dict" in ck:
            ck["state_dict"] = sd

        # Save the modified checkpoint
        torch.save(ck, file)
        log.warning(f"Saved patched checkpoint to {file}")
        
        # Create a "patched" file when patching is done
        patched.touch()
            

    def unpatch_model_430(self, file: Path):
        """
        Since CLAP 1.1.6, its codebase provided its own workarounds that isn't compatible
        with our patch. This function will revert the patch to make it compatible with the new
        CLAP version.
        """
        patched = file.parent / f"{file.name}.patched.430"
        if not patched.exists():
            return
        
        # The below is an inverse operation of the patch_model_430 function, so comments are omitted
        log.warning("Unpatching LAION-CLAP's model checkpoints")
        ck = torch.load(file, map_location="cpu")
        unwrap = isinstance(ck, dict) and "state_dict" in ck
        sd = ck["state_dict"] if unwrap else ck
        sd["module.text_branch.embeddings.position_ids"] = 0
        if isinstance(ck, dict) and "state_dict" in ck:
            ck["state_dict"] = sd
        torch.save(ck, file)
        log.warning(f"Saved unpatched checkpoint to {file}")
        patched.unlink()
        
        
    def load_model(self):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny' if self.type == 'audio' else 'HTSAT-base')
        self.model.load_ckpt(self.model_file)
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
        emb = torch.cat(embeddings, dim=0) # [timeframes, 512]
        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)


class CdpamModel(ModelLoader):
    """
    CDPAM model from https://github.com/pranaymanocha/PerceptualAudio/tree/master/cdpam
    """
    def __init__(self, mode: Literal['acoustic', 'content']) -> None:
        super().__init__(f"cdpam-{mode}", 512, 22050)
        self.mode = mode
        assert mode in ['acoustic', 'content'], "Mode must be 'acoustic' or 'content'"

    def load_model(self):
        from cdpam import CDPAM
        self.model = CDPAM(dev=self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        audio = torch.from_numpy(audio).float().to(self.device)

        # Take 1s chunks
        chunk_size = self.sr
        frames = []
        for i in range(0, audio.shape[1], chunk_size):
            chunk = audio[:, i:i+chunk_size]
            _, acoustic, content = self.model.model.base_encoder.forward(chunk.unsqueeze(1))
            v = acoustic if self.mode == 'acoustic' else content
            v = F.normalize(v, dim=1)
            frames.append(v)

        # Concatenate the embeddings
        emb = torch.cat(frames, dim=0) # [timeframes, 512]
        return emb

    def load_wav(self, wav_file: Path):
        x, _  = librosa.load(wav_file, sr=self.sr)
        
        # Convert to 16 bit floating point
        x = np.round(x.astype(np.float) * 32768)
        x  = np.reshape(x, [-1, 1])
        x = np.reshape(x, [1, x.shape[0]])
        x  = np.float32(x)
        
        return x


class CLAPModel(ModelLoader):
    """
    CLAP model from https://github.com/microsoft/CLAP
    """
    def __init__(self, type: Literal['2023']):
        super().__init__(f"clap-{type}", 1024, 44100)
        self.type = type

        if type == '2023':
            url = 'https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth'

        self.model_file = Path(__file__).parent / ".model-checkpoints" / url.split('/')[-1]

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)

    def load_model(self):
        from msclap import CLAP
        
        self.model = CLAP(self.model_file, version = self.type, use_cuda=self.device == torch.device('cuda'))
        #self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.reshape(1, -1)

        # The int16-float32 conversion is used for quantization
        #audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 7s chunks with 1s hop
        chunk_size = 7 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # zero-pad chunks to make equal length
        clen = [x.shape[1] for x in chunks]
        chunks = [np.pad(ch, ((0,0), (0,np.max(clen) - ch.shape[1]))) for ch in chunks]

        self.model.default_collate(chunks)

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.clap.audio_encoder(chunk)[0]
                embeddings.append(emb)

        # Concatenate the embeddings
        emb = torch.cat(embeddings, dim=0) # [timeframes, 1024]
        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)


class W2V2Model(ModelLoader):
    """
    W2V2 model from https://huggingface.co/facebook/wav2vec2-base-960h, https://huggingface.co/facebook/wav2vec2-large-960h

    Please specify the size ('base' or 'large') and the layer to use (1-12 for 'base' or 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'large'], layer: Literal['12', '24'], limit_minutes=6):
        model_dim = 768 if size == 'base' else 1024
        model_identifier = f"w2v2-{size}" + ("" if (layer == 12 and size == 'base') or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000)
        self.huggingface_id = f"facebook/wav2vec2-{size}-960h"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, Wav2Vec2Model
        
        self.model = Wav2Vec2Model.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]

        return out


class HuBERTModel(ModelLoader):
    """
    HuBERT model from https://huggingface.co/facebook/hubert-base-ls960, https://huggingface.co/facebook/hubert-large-ls960

    Please specify the size ('base' or 'large') and the layer to use (1-12 for 'base' or 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'large'], layer: Literal['12', '24'], limit_minutes=6):
        model_dim = 768 if size == 'base' else 1024
        model_identifier = f"hubert-{size}" + ("" if (layer == 12 and size == 'base') or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000)
        self.huggingface_id = f"facebook/hubert-{size}-ls960"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, HubertModel

        self.model = HubertModel.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]

        return out


class WavLMModel(ModelLoader):
    """
    WavLM model from https://huggingface.co/microsoft/wavlm-base, https://huggingface.co/microsoft/wavlm-base-plus, https://huggingface.co/microsoft/wavlm-large

    Please specify the model size ('base', 'base-plus', or 'large') and the layer to use (1-12 for 'base' or 'base-plus' and 1-24 for 'large').
    """
    def __init__(self, size: Literal['base', 'base-plus', 'large'], layer: Literal['12', '24'], limit_minutes=6):
        model_dim = 768 if size in ['base', 'base-plus'] else 1024
        model_identifier = f"wavlm-{size}" + ("" if (layer == 12 and size in ['base', 'base-plus']) or (layer == 24 and size == 'large') else f"-{layer}")

        super().__init__(model_identifier, model_dim, 16000)
        self.huggingface_id = f"patrickvonplaten/wavlm-libri-clean-100h-{size}"
        self.layer = layer
        self.limit = limit_minutes * 60 * self.sr

    def load_model(self):
        from transformers import AutoProcessor, WavLMModel

        self.model = WavLMModel.from_pretrained(self.huggingface_id)
        self.processor = AutoProcessor.from_pretrained(self.huggingface_id)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Limit to specified minutes
        if audio.shape[0] > self.limit:
            log.warning(f"Audio is too long ({audio.shape[0] / self.sr / 60:.2f} minutes > {self.limit / self.sr / 60:.2f} minutes). Truncating.")
            audio = audio[:self.limit]

        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            out = torch.stack(out.hidden_states).squeeze()  # [13 or 25 layers, timeframes, 768 or 1024]
            out = out[self.layer]  # [timeframes, 768 or 1024]

        return out


class WhisperModel(ModelLoader):
    """
    Whisper model from https://huggingface.co/openai/whisper-base
    
    Please specify the model size ('tiny', 'base', 'small', 'medium', or 'large').
    """
    def __init__(self, size: Literal['tiny', 'base', 'small', 'medium', 'large']):
        dimensions = {
            'tiny': 384,
            'base': 512,
            'small': 768,
            'medium': 1024,
            'large': 1280
        }
        model_dim = dimensions.get(size)
        model_identifier = f"whisper-{size}"

        super().__init__(model_identifier, model_dim, 16000)
        self.huggingface_id = f"openai/whisper-{size}"
        
    def load_model(self):
        from transformers import AutoFeatureExtractor
        from transformers import WhisperModel
        
        self.model = WhisperModel.from_pretrained(self.huggingface_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.huggingface_id)
        self.decoder_input_ids = (torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id).to(self.device)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(audio, sampling_rate=self.sr, return_tensors="pt").to(self.device)
        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            out = self.model(input_features, decoder_input_ids=self.decoder_input_ids).last_hidden_state  # [1, timeframes, 512]
            out = out.squeeze() # [timeframes, (384 or 512 or 768 or 1024 or 1280)]

        return out



def get_all_models() -> list[ModelLoader]:
    ms = [
        CLAPModel('2023'),
        CLAPLaionModel('audio'), CLAPLaionModel('music'),
        VGGishModel(), 
        *(MERTModel(layer=v) for v in range(1, 13)),
        EncodecEmbModel('24k'), EncodecEmbModel('48k'), 
        # DACModel(),
        # CdpamModel('acoustic'), CdpamModel('content'),
        *(W2V2Model('base', layer=v) for v in range(1, 13)),
        *(W2V2Model('large', layer=v) for v in range(1, 25)),
        *(HuBERTModel('base', layer=v) for v in range(1, 13)),
        *(HuBERTModel('large', layer=v) for v in range(1, 25)),
        *(WavLMModel('base', layer=v) for v in range(1, 13)),
        *(WavLMModel('base-plus', layer=v) for v in range(1, 13)),
        *(WavLMModel('large', layer=v) for v in range(1, 25)),
        WhisperModel('tiny'), WhisperModel('small'),
        WhisperModel('base'), WhisperModel('medium'),
        WhisperModel('large'),
    ]
    if importlib.util.find_spec("dac") is not None:
        ms.append(DACModel())
    if importlib.util.find_spec("cdpam") is not None:
        ms += [CdpamModel('acoustic'), CdpamModel('content')]

    return ms
