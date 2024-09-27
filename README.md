# Frechet Audio Distance Toolkit

A simple and standardized library for Frechet Audio Distance (FAD) calculation. This library is published along with the paper [_Adapting Frechet Audio Distance for Generative Music Evaluation_](https://ieeexplore.ieee.org/document/10446663) ([arXiv](https://arxiv.org/abs/2311.01616)). The datasets associated with this paper and sample code tools used in the paper are also available under this repository.

You can listen to audio samples of per-song FAD outliers on the online demo here: https://fadtk.hydev.org/

## 0x00. Features

* Easily and efficiently compute audio embeddings with various models.
* Compute FAD∞ scores between two datasets for evaluation.
* Use pre-computed statistics ("weights") to compute FAD∞ scores from existing baselines.
* Compute per-song FAD to find outliers in the dataset

### Supported Models

| Model | Name in FADtk | Description | Creator |
| --- | --- | --- | --- |
| [CLAP](https://github.com/microsoft/CLAP) | `clap-2023` | Learning audio concepts from natural language supervision | Microsoft |
| [CLAP](https://github.com/LAION-AI/CLAP) | `clap-laion-{audio/music}` | Contrastive Language-Audio Pretraining | LAION |
| [Encodec](https://github.com/facebookresearch/encodec) | `encodec-emb` | State-of-the-art deep learning based audio codec | Facebook/Meta Research |
| [MERT](https://huggingface.co/m-a-p/MERT-v1-95M) | `MERT-v1-95M-{layer}` | Acoustic Music Understanding Model with Large-Scale Self-supervised Training | m-a-p |
| [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md) | `vggish` | Audio feature classification embedding | Google |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | `dac-44kHz` | High-Fidelity Audio Compression with Improved RVQGAN | Descript |
| [CDPAM](https://github.com/pranaymanocha/PerceptualAudio) | `cdpam-{acoustic/content}` | Contrastive learning-based Deep Perceptual Audio Metric | Pranay Manocha et al. |
| [Wav2vec 2.0](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) | `w2v2-{base/large}` | Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Facebook/Meta Research |
| [HuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md) | `hubert-{base/large}` | HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | Facebook/Meta Research |
| [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) | `wavlm-{base/base-plus/large}` | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | Microsoft |
| [Whisper](https://github.com/openai/whisper) | `whisper-{tiny/base/small/medium/large}` | Robust Speech Recognition via Large-Scale Weak Supervision | OpenAI |

## 0x01. Installation

To use the FAD toolkit, you must first install it. This library is created and tested on Python 3.11 on Linux but should work on Python >3.9 and on Windows and macOS as well.

1. Install torch https://pytorch.org/
2. `pip install fadtk`

To ensure that the environment is setup correctly and everything work as intended, it is recommended to run our tests using the command `python -m fadtk.test` after installing.

### Optional Dependencies

Optionally, you can install dependencies that add additional embedding support. They are:

* CDPAM: `pip install cdpam`
* DAC: `pip install descript-audio-codec==1.0.0`

## 0x02. Command Line Usage

```sh
# Evaluation
fadtk <model_name> <baseline> <evaluation-set> [--inf/--indiv]

# Compute embeddings
fadtk.embeds -m <models...> -d <datasets...>
```
#### Example 1: Computing FAD_inf scores on FMA_Pop baseline
  
```sh
# Compute FAD-inf between the baseline and evaluation datasets on two different models
fadtk clap-laion-audio fma_pop /path/to/evaluation/audio --inf
fadtk encodec-emb fma_pop /path/to/evaluation/audio --inf
```

#### Example 2: Compute individual FAD scores for each song

```sh
fadtk encodec-emb fma_pop /path/to/evaluation/audio --indiv scores.csv
```

#### Example 3: Compute FAD scores with your own baseline

First, create two directories, one for the baseline and one for the evaluation, and place *only* the audio files in them. Then, run the following commands:

```sh
# Compute FAD between the baseline and evaluation datasets
fadtk clap-laion-audio /path/to/baseline/audio /path/to/evaluation/audio
```

#### Example 4: Just compute embeddings

If you only want to compute embeddings with a list of specific models for a list of dataset, you can do that using the command line.

```sh
fadtk.embeds -m Model1 Model2 -d /dataset1 /dataset2
```

## 0x03. Best Practices

When using the FAD toolkit to compute FAD scores, it's essential to consider the following best practices to ensure accuracy and relevancy in the reported findings.

1. **Choose a Meaningful Reference Set**: Do not default to commonly used reference sets like Musiccaps without consideration. A reference set that aligns with the specific goal of the research should be chosen. For generative music, we recommend using the FMA-Pop subset as proposed in our paper.
2. **Select an Appropriate Embedding**: The choice of embedding can heavily influence the scoring. For instance, VGGish is optimized for classification, and it might not be the most suitable if the research objective is to measure aspects like quality.
3. **Provide Comprehensive Reporting**: Ensure that all test statistics are included in the report:
   * The chosen reference set.
   * The selected embedding.
   * The number of samples and their duration in both the reference and test set.
     
   This level of transparency ensures that the FAD scores' context and potential variability are understood by readers or users.
4. **Benchmark Against the State-of-the-Art**: When making comparisons, researchers should ideally use the same setup to assess the state-of-the-art models for comparison. Without a consistent setup, the FAD comparison might lose its significance.
5. **Interpret FAD Scores Contextually**: Per-sample FAD scores should be calculated. Listening to the per-sample outliers will provide a hands-on understanding of what the current setup is capturing, and what "low" and "high" FAD scores signify in the context of the study.

By adhering to these best practices, the use of our FAD toolkit can be ensured to be both methodologically sound and contextually relevant.


## 0x04. Programmatic Usage

### Doing the above in python

If you want to know how to do the above command-line processes in python, you can check out how our launchers are implemented ([\_\_main\_\_.py](fadtk/__main__.py) and [embeds.py](fadtk/embeds.py))

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

## 0x05. Published Data and Code

We also include some sample code and data from the paper in this repo.

### Refined Datasets

[musiccaps-public-openai.csv](datasets/musiccaps-public-openai.csv): This file contains the original MusicCaps song IDs and captions along with GPT4 labels for their quality and the GPT4-refined prompts used for music generation.

[fma_pop_tracks.csv](datasets/fma_pop_tracks.csv): This file contains the subset of 4839 song IDs and metadata information for the FMA-Pop subset we proposed in our paper. After downloading the Free Music Archive dataset from the [original source](https://github.com/mdeff/fma), you can easily locate the audio files for this FMA-Pop subset using song IDs.

### Sample Code

The method we used to create GPT4 one-shot prompts for generating the refined MusicCaps prompts and for classifying quality from the MusicCaps captions can be found in [example/prompts](example/prompts).

## 0x06. Citation, Acknowledgments and Licenses

The code in this toolkit is licensed under the [MIT License](./LICENSE). Please cite our work if this repository helped you in your project. 

```latex
@inproceedings{fadtk,
  title = {Adapting Frechet Audio Distance for Generative Music Evaluation},
  author = {Azalea Gui, Hannes Gamper, Sebastian Braun, Dimitra Emmanouilidou},
  booktitle = {Proc. IEEE ICASSP 2024},
  year = {2024},
  url = {https://arxiv.org/abs/2311.01616},
}
```

Please also cite the FMA (Free Music Archive) dataset if you used FMA-Pop as your FAD scoring baseline.

```latex
@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
```

You may also refer to our work on measuring music emotion and mitigating emotion bias using this toolkit.

```latex
@article{emotionbias_fad,
  title = {Rethinking Emotion Bias in Music via Frechet Audio Distance},
  author = {Li, Yuanchao and Gui, Azalea and Emmanouilidou, Dimitra and Gamper, Hannes},
  journal={arXiv preprint arXiv:2409.15545},
  year={2024}
}
```

### Special Thanks

**Immense gratitude to the foundational repository [gudgud96/frechet-audio-distance](https://github.com/gudgud96/frechet-audio-distance) - "A lightweight library for Frechet Audio Distance calculation"**. Much of our project has been adapted and enhanced from gudgud96's contributions. In honor of this work, we've retained the [original MIT license](example/LICENSE_gudgud96).

* Encodec from Facebook: [facebookresearch/encodec](https://github.com/facebookresearch/encodec/)
* CLAP: [microsoft/CLAP](https://github.com/microsoft/CLAP)
* CLAP from LAION: [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
* MERT from M-A-P: [m-a-p/MERT](https://huggingface.co/m-a-p/MERT-v1-95M)
* Wav2vec 2.0: [facebookresearch/wav2vec 2.0](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)
* HuBERT: [facebookresearch/HuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md)
* WavLM: [microsoft/WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)
* Whisper: [OpenAI/Whisper](https://github.com/openai/whisper)
* VGGish in PyTorch: [harritaylor/torchvggish](https://github.com/harritaylor/torchvggish)
* Free Music Archive: [mdeff/fma](https://github.com/mdeff/fma)
* Frechet Inception Distance implementation: [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
* Frechet Audio Distance paper: [arxiv/1812.08466](https://arxiv.org/abs/1812.08466)
