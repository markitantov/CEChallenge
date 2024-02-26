#!/bin/bash
source "abaw_env/bin/activate"

jupyter notebook --port=8888 --NotebookApp.token='' --notebook-dir ./notebooks/ --NotebookApp.iopub_data_rate_limit=1e10
