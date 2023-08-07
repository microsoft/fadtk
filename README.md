## FAD Toolkit

A simple and standardized library for Frechet Audio Distance (FAD) calculation.

### Installation

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

### Thanks to

Original FAD implementation: https://github.com/gudgud96/frechet-audio-distance

VGGish in PyTorch: https://github.com/harritaylor/torchvggish

Frechet distance implementation: https://github.com/mseitzer/pytorch-fid

Frechet Audio Distance paper: https://arxiv.org/abs/1812.08466