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

    @abstractmethod
    def load_model(self):
        pass

    def get_embedding(self, audio: np.ndarray):
        embd = self.__get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        return embd

    @abstractmethod
    def __get_embedding(self, audio: np.ndarray):
        pass
