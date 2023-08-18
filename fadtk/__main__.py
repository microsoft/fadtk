from argparse import ArgumentParser

from .fad import FrechetAudioDistance, log
from .model_loader import *
from .fad_batch import cache_embedding_files

if __name__ == "__main__":
    """
    Launcher for running FAD on two directories using a model.
    """
    models = {m.name: m for m in get_all_models()}

    agupa = ArgumentParser()
    # Two positional arguments: model and two directories
    agupa.add_argument('model', type=str, choices=list(models.keys()), help="The embedding model to use")
    agupa.add_argument('baseline', type=str, help="The baseline dataset")
    agupa.add_argument('eval', type=str, help="The directory to evaluate against")

    # Add optional arguments
    agupa.add_argument('-w', '--workers', type=int, default=8)
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox')
    agupa.add_argument('--inf', action='store_true', help="Use FAD-inf extrapolation")

    args = agupa.parse_args()
    model = models[args.model]

    baseline = Path(args.baseline)
    eval = Path(args.eval)

    # 1. Calculate embedding files for each dataset
    for d in [baseline, eval]:
        if d.is_dir():
            cache_embedding_files(d, model, workers=args.workers, sox_path=args.sox_path)
    
    # 2. Calculate FAD
    fad = FrechetAudioDistance(model, audio_load_worker=args.workers, sox_path=args.sox_path, load_model=False)
    if args.inf:
        assert eval.is_dir(), "FAD-inf requires a directory as the evaluation dataset"
        score = fad.score_inf(baseline, eval)
    else:
        score = fad.score(baseline, eval)

    # 3. Print results    
    log.info("FAD computed.")
    log.info(f"The FAD {model.name} score between {baseline} and {eval} is: {score}")
    
