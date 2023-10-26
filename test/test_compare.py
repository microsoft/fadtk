import os
from pathlib import Path
import numpy as np
import pandas as pd

import fadtk
from fadtk.fad import FrechetAudioDistance
from fadtk.fad_batch import cache_embedding_files
from fadtk.model_loader import get_all_models


if __name__ == '__main__':
    # Read samples csv
    fp = Path(__file__).parent
    reference = pd.read_csv(fp / 'samples_FAD_scores.csv')

    # Compute FAD score
    for model in get_all_models():
        print(f'Computing FAD score for {model.name}')
        csv = fp / 'fad_scores' / f'{model.name}.csv'
        if csv.is_file():
            continue
        
        fad = FrechetAudioDistance(model, audio_load_worker=1, load_model=True)
        
        # Cache embedding files
        # cache_embedding_files(fp / 'samples', model, workers=4)
        for f in (fp / 'samples').glob('*.*'):
            fad.cache_embedding_file(f)
        
        # Compute FAD score
        fad.score_individual('fma_pop', fp / 'samples', csv)

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
        table.append({
            'model': model_name,
            'mse': mse,
            'max_abs_diff': max_abs_diff,
            'mean': mean,
            'mad%': max_abs_diff / mean * 100
        })
        
    # Print table
    table = pd.DataFrame(table)
    print(table)
    table.to_csv(fp / 'comparison.csv')
        
        
    
