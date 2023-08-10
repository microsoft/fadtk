from argparse import ArgumentParser

from .fad import FrechetAudioDistance, log
from .model_loader import *
from .fad_batch import cache_embeddings_files

if __name__ == "__main__":
    """
    Launcher for running FAD on two directories using a model.
    """
    models = {m.name: m for m in get_all_models()}

    agupa = ArgumentParser()
    # Two positional arguments: model and two directories
    agupa.add_argument('model', type=str, choices=list(models.keys()))
    agupa.add_argument('baseline_dir', type=str)
    agupa.add_argument('eval_dir', type=str)

    # Add optional arguments
    agupa.add_argument('-w', '--workers', type=int, default=8)
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox')

    args = agupa.parse_args()
    model = models[args.model]

    fad = FrechetAudioDistance(model, audio_load_worker=args.workers, sox_path=args.sox_path, load_model=False)
    log.info("FAD computed.")
    log.info(f"The FAD {model.name} score between {args.baseline_dir} and {args.eval_dir} is: {fad.score(args.baseline_dir, args.eval_dir)}")
    
