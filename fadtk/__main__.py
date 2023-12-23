import time
from argparse import ArgumentParser

from .fad import FrechetAudioDistance, log
from .model_loader import *
from .fad_batch import cache_embedding_files


def main():
    """
    Launcher for running FAD on two directories using a model.
    """
    models = {m.name: m for m in get_all_models()}

    agupa = ArgumentParser()
    # Two positional arguments: model and two directories
    agupa.add_argument('model', type=str, choices=list(models.keys()), help="The embedding model to use")
    agupa.add_argument('baseline', type=str, help="The baseline dataset")
    agupa.add_argument('eval', type=str, help="The directory to evaluate against")
    agupa.add_argument('csv', type=str, nargs='?',
                       help="The CSV file to append results to. "
                            "If this argument is not supplied, single-value results will be printed to stdout, "
                            "and for --indiv, the results will be saved to 'fad-individual-results.csv'")

    # Add optional arguments
    agupa.add_argument('-w', '--workers', type=int, default=8)
    agupa.add_argument('-s', '--sox-path', type=str, default='/usr/bin/sox')
    agupa.add_argument('--inf', action='store_true', help="Use FAD-inf extrapolation")
    agupa.add_argument('--indiv', action='store_true',
                       help="Calculate FAD for individual songs and store the results in the given file")

    args = agupa.parse_args()
    model = models[args.model]

    baseline = args.baseline
    eval = args.eval

    # 1. Calculate embedding files for each dataset
    for d in [baseline, eval]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=args.workers)

    # 2. Calculate FAD
    fad = FrechetAudioDistance(model, audio_load_worker=args.workers, load_model=False)
    if args.inf:
        assert Path(eval).is_dir(), "FAD-inf requires a directory as the evaluation dataset"
        score = fad.score_inf(baseline, list(Path(eval).glob('*.*')))
        print("FAD-inf Information:", score)
        score, inf_r2 = score.score, score.r2
    elif args.indiv:
        assert Path(eval).is_dir(), "Individual FAD requires a directory as the evaluation dataset"
        csv_path = Path(args.csv or 'fad-individual-results.csv')
        fad.score_individual(baseline, eval, csv_path)
        log.info(f"Individual FAD scores saved to {csv_path}")
        exit(0)
    else:
        score = fad.score(baseline, eval)
        inf_r2 = None

    # 3. Print results    
    log.info("FAD computed.")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        if not Path(args.csv).is_file():
            Path(args.csv).write_text('model,baseline,eval,score,inf_r2,time\n')
        with open(args.csv, 'a') as f:
            f.write(f'{model.name},{baseline},{eval},{score},{inf_r2},{time.time()}\n')
        log.info(f"FAD score appended to {args.csv}")

    log.info(f"The FAD {model.name} score between {baseline} and {eval} is: {score}")


if __name__ == "__main__":
    main()
