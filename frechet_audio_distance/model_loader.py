from abc import ABC, abstractmethod
import os
import numpy as np

import torch
from torch import nn

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
        pass

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

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        return self.model.forward(audio, self.sr)

class PANNModel(ModelLoader):
    """
    Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020)
    """
    def __init__(self):
        super().__init__("pann")

    def load_model(self):
        model_path = os.path.join(torch.hub.get_dir(), "Cnn14_16k_mAP%3D0.438.pth")
        if not(os.path.exists(model_path)):
            torch.hub.download_url_to_file('https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth', torch.hub.get_dir())
        self.model = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.sr = 16000
        self.model.eval()

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.model(torch.tensor(audio).float().unsqueeze(0), None)
            return out['embedding'].data[0]
