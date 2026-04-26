#!/bin/sh

dvc config core.analytics false
dvc init -f
dvc remote add -d myremote datasets -f
dvc add energydata_complete_v1.csv && dvc commit
git add energydata_complete_v1.csv.dvc .dvc/config && git commit -m "initial_data"
git add params.yaml preprocessing.py train.py dvc.yaml dvc.lock
git commit -m "added core files"
dvc push
dvc exp remove -A
dvc queue remove --all
touch dvc.yaml
git checkout -b dvc-241106
dvc exp save --name first -m "241106"
dvc exp show
dvc stage add --name=preprocessing --force python preprocessing.py
dvc stage add --name=postprocessing --force python postprocessing.py
dvc stage add --name prepare --force python prepare.py
dvc stage add --name train --force python train.py