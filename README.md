# Frechet Audio Distance Toolkit

A simple and standardized library for Frechet Audio Distance (FAD) calculation. This library is published along with the paper _Improving Frechet Audio Distance for Generative Music Evaluation_ (link coming soon). The datasets associated with this paper and sample code tools used in the paper are also available under this repository.

## 0x01. Toolkit Usage

To use the FAD toolkit, you must first install it. This library is tested on Python 3.11 on Linux but should work on Python >3.9 and on Windows as well.

`pip install fadtk`

### Command Line Usage

First, create two directories, one for the baseline and one for the evaluation, and place *only* the audio files in them. Then, run the following commands:

```sh
# Compute embeddings for the baseline dataset
python3 -m fadtk.embds MODEL-NAME /path/to/baseline/audio
# Compute embeddings for the evaluation dataset (test set)
python3 -m fadtk.embds MODEL-NAME /path/to/evaluation/audio
# Compute FAD between the baseline and evaluation datasets
python3 -m fadtk.score MODEL-NAME /path/to/baseline/audio /path/to/evaluation/audio
```

### Adding New Embeddings

To add a new embedding, the only file you would need to modify is [model_loader.py](fadtk/model_loader.py). You must create a new class that inherits the ModelLoader class. You need to implement the constructor, the `load_model` and the `_get_embedding` function. You can start with the below template:

```python
class YourModel(ModelLoader):
    """
    Add a short description of your model here.
    """
    def __init__(self):
        # Define your sample rate and number of features here. Audio will automatically be resampled to this sample rate.
        super().__init__("Model name including variant", num_features=128, sr=16000)
        # Add any other variables you need here

    def load_model(self):
        # Load your model here
        pass

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        # Calculate the embeddings using your model
        return np.zeros((1, self.num_features))

    def load_wav(self, wav_file: Path):
        # Optionally, you can override this method to load wav file in a different way. The input wav_file is already in the correct sample rate specified in the constructor.
        return super().load_wav(wav_file)
```

## 0x02. Published Data and Code

We also include some sample code and data from the paper in this repo.

### Refined Datasets

[musiccaps-public-openai.csv](datasets/musiccaps-public-openai.csv): This file contains the original MusicCaps song IDs and captions along with GPT4 labels for their quality and the GPT4-refined prompts used for music generation.

* The method we used to create GPT4 one-shot prompts for these generation can be found in [example/prompts](example/prompts).

[fma_pop_tracks.csv](): This file contains the subset of 4839 song IDs and metadata information for the FMA-Pop subset we proposed in our paper. After downloading the Free Music Archive dataset from the [original source](https://github.com/mdeff/fma), you can easily locate the audio files for this FMA-Pop subset using song IDs.

### Sample Code


## 0x03. Special Thanks

**Immense gratitude to the foundational repository [gudgud96/frechet-audio-distance](https://github.com/gudgud96/frechet-audio-distance) - "A lightweight library for Frechet Audio Distance calculation"**. Much of our project has been adapted and enhanced from gudgud96's contributions. In honor of this work, we've retained the [original MIT license](example/LICENSE_gudgud96).

* Encodec from Facebook: [facebookresearch/encodec](https://github.com/facebookresearch/encodec/)
* CLAP from LAION: [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
* MERT from M-A-P: [m-a-p/MERT](m-a-p/MERT-v1-95M) 
* VGGish in PyTorch: [harritaylor/torchvggish](https://github.com/harritaylor/torchvggish)
* Free Music Archive: [mdeff/fma](https://github.com/mdeff/fma)
* Frechet Inception Distance implementation: [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
* Frechet Audio Distance paper: [arxiv/1812.08466](https://arxiv.org/abs/1812.08466)