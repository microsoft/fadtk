from argparse import ArgumentParser
from .model_loader import *
from .fad_batch import cache_embeddings_files

if __name__ == "__main__":
    """
    Launcher for caching embeddings of a directory using a model.
    """
    models = [
        CLAPLaionModel('audio'), CLAPLaionModel('music'),
        VGGishModel(), 
        *(MERTModel(layer=v) for v in range(1, 13)),
        EncodecEmbModel('24k'), EncodecEmbModel('48k'), 
        DACModel(),
        CdpamModel('acoustic'), CdpamModel('content'),
    ]
    models = {m.name: m for m in models}

    agupa = ArgumentParser()
    # Two positional arguments: model and directory
    agupa.add_argument('model', type=str, choices=list(models.keys()))
    agupa.add_argument('directory', type=str)

    # Add optional arguments
    agupa.add_argument('-w', '--workers', type=int, default=8)
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox')

    args = agupa.parse_args()
    model = models[args.model]

    cache_embeddings_files(args.directory, model, workers=args.workers, sox_path=args.sox_path)
    
