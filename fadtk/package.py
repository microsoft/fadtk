from argparse import ArgumentParser

from .fad import FrechetAudioDistance
from .model_loader import *
from .fad_batch import cache_embedding_files

if __name__ == "__main__":
    """
    Launcher for packaging statistics of a directory using a model.
    """
    models = {m.name: m for m in get_all_models()}

    agupa = ArgumentParser()
    agupa.add_argument('directory', type=str)
    agupa.add_argument('out', type=str)

    # Add optional arguments
    agupa.add_argument('-w', '--workers', type=int, default=8)
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox')

    args = agupa.parse_args()

    out = Path(args.out)
    if out.suffix != '.npz':
        print('The output file you specified is not a npz file, are you sure? (y/N)')
        if input().lower() != 'y':
            exit(1)

    # 1. Calculate embedding files for each model
    for model in models.values():
        cache_embedding_files(args.directory, model, workers=args.workers)
    
    # 2. Calculate statistics for each model
    data = {}
    for model in models.values():
        fad = FrechetAudioDistance(model, load_model=False)
        mu, cov = fad.load_stats(args.directory)
        data[f'{model.name}.mu'] = mu
        data[f'{model.name}.cov'] = cov

    # 3. Save statistics
    np.savez(out, **data)