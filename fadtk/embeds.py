from argparse import ArgumentParser
from .model_loader import *
from .fad_batch import cache_embedding_files

def main():
    """
    Launcher for caching embeddings of directories using multiple models.
    """
    models = {m.name: m for m in get_all_models()}

    agupa = ArgumentParser()
    
    # Accept multiple models and directories with distinct prefixes
    agupa.add_argument('-m', '--models', type=str, choices=list(models.keys()), nargs='+', required=True)
    agupa.add_argument('-d', '--dirs', type=str, nargs='+', required=True)

    # Add optional arguments
    agupa.add_argument('-w', '--workers', type=int, default=8)
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox')

    args = agupa.parse_args()

    for model_name in args.models:
        model = models[model_name]
        for d in args.dirs:
            log.info(f"Caching embeddings for {d} using {model.name}")
            cache_embedding_files(d, model, workers=args.workers)

            
if __name__ == "__main__":
    main()