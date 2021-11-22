#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/process_data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')

from etl import get_data
from preprocess import save_cleaned_corpus
from lda import save_lda_model


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'etl' in targets:
        with open('config/etl-params.json') as fh:
            etl_cfg = json.load(fh)

        # make the data target
        data = get_data(**etl_cfg)


    if 'process_data' in targets:
        with open('config/process-params.json') as fh:
            process_cfg = json.load(fh)

        save_cleaned_corpus(**process_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        save_lda_model(**model_cfg)

    if 'test' in targets:
        with open('config/test-process-params.json') as fh:
            process_cfg = json.load(fh)
        save_cleaned_corpus(**process_cfg)

        with open('config/test-model-params.json') as fh:
            model_cfg = json.load(fh)
        save_lda_model(**model_cfg)

    return


if __name__ == '__main__':
    # run via:
    # python main.py
    targets = sys.argv[1:]
    main(targets)
