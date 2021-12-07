# Faculty Retrieval Information System

This repository contains project code for experimenting with LDA for Faculty Information Retrieval System.

## Running the Project
* To get the preprocessed data file, run `python run.py process_data`
* To get the fitted sklearn.lda model, run `python run.py model`
* To test the model, run `python run.py test`
* To prepare/update the dashboard, run `python run.py prepare_sankey`
* To run the live dashboard, run `python run.py run_dashboard`

## Using the Dashboard
* When executing `test` or `run_dashboard`, it will launch dash with a locally hosted port.
* It would require port-forwarding on a remote server.
