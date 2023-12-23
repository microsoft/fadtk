from pathlib import Path
import traceback
import numpy as np
import pandas as pd

from fadtk.fad import FrechetAudioDistance
from fadtk.model_loader import get_all_models
from hypy_utils.logging_utils import setup_logger

log = setup_logger()

if __name__ == '__main__':
    # Read samples csv
    fp = Path(__file__).parent
    reference = pd.read_csv(fp / 'samples_FAD_scores.csv')

    # Get reference models in column names
    reference_models = [c.split('_', 1)[1].replace('_fma_pop', '') for c in reference.columns if c.startswith('FAD_')]
    print("Models with reference data:", reference_models)

    # Compute FAD score
    for model in get_all_models():
        if model.name.replace('-', '_') not in reference_models:
            print(f'No reference data for {model.name}, skipping')
            continue

        # Because of the heavy computation required to run each test, we limit the MERT models to only a few layers
        if model.name.startswith('MERT') and model.name[-1] not in ['1', '4', '8', 'M']:
            continue

        log.info(f'Computing FAD score for {model.name}')
        csv = fp / 'fad_scores' / f'{model.name}.csv'
        if csv.is_file():
            continue
        
        fad = FrechetAudioDistance(model, audio_load_worker=1, load_model=True)
        
        # Cache embedding files
        try:
            for f in (fp / 'samples').glob('*.*'):
                fad.cache_embedding_file(f)
        except Exception as e:
            traceback.print_exc()
            log.error(f'Error when caching embedding files for {model.name}: {e}')
            exit(1)
        
        try:
            # Compute FAD score
            fad.score_individual('fma_pop', fp / 'samples', csv)
        except Exception as e:
            traceback.print_exc()
            log.error(f'Error when computing FAD score for {model.name}: {e}')
            exit(1)
            
        # Compute FAD for entire set
        all_score = fad.score('fma_pop', fp / 'samples')
        
        # Add all_score to csv with file name '/samples/all'
        data = pd.read_csv(csv, names=['file', 'score'])
        data = pd.concat([data, pd.DataFrame([['/samples/all', all_score]], columns=['file', 'score'])])
        data.to_csv(csv, index=False, header=False)

    # Read from csvs
    table = []
    for f in (fp / 'fad_scores').glob('*.csv'):
        model_name = f.stem.replace('-', '_')
        data = pd.read_csv(f, names=['file', 'score'])
        data['file'] = data['file'].replace(r'\\', '/', regex=True) # convert Windows paths
        data['file'] = data['file'].apply(lambda x: '/'.join(x.split('/')[-2:]).split('.')[0])
        
        # Get the scores of the same model from the reference csv as an array
        # They should be in FAD_{model_name}_fma_pop column
        test = reference.loc[:, ['song_id', f'FAD_{model_name}_fma_pop']].copy()
        test.columns = ['file', 'score']
        
        # Transform test to a dictionary of file: score
        test = test.set_index('file').to_dict()['score']
        
        test = np.array([test[f] for f in data['file']])
        data = np.array(data['score'])
        
        # Compare mean sqaurred error
        mse = ((data - test) ** 2).mean()
        max_abs_diff = np.abs(data - test).max()
        mean = np.mean(data)
        madp = max_abs_diff / mean * 100
        table.append({
            'model': model_name,
            'mse': mse,
            'max_abs_diff': max_abs_diff,
            'mean': mean,
            'mad%': madp,
            'pass': madp < 5  # 5% threshold
        })
        
    # Print table
    table = pd.DataFrame(table)
    log.info(table)
    table.to_csv(fp / 'comparison.csv')
    
    # If anything failed, exit with error code 2
    if not table['pass'].all():
        log.error('Some models failed the test')
        exit(2)
