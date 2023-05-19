#!bin/sh

# Interactive
pjsub --interact -j -o interactive.out -g gk77 -L rscgrp=interactive-a,node=1,jobenv=singularity
module load singularity/3.7.3

# Scraping
singularity shell --pwd $HOME/Instance_level_recognition/app/scraping/src/ --nv $HOME/Instance_level_recognition/app/scraping/scraping_latest.sif

# Object Detection
singularity shell --pwd $HOME/Instance_level_recognition/app/object_detection/src/ --nv $HOME/Instance_level_recognition/app/object_detection/object_detection_latest.sif
singularity shell --pwd $HOME/Instance_level_recognition/app/object_detection/src/ --nv $HOME/Instance_level_recognition/app/ml/python_ml_latest.sif 

# ML
module load singularity/3.7.3
singularity shell --pwd $HOME/Instance_level_recognition/app/ml/src/ --nv $HOME/Instance_level_recognition/app/ml/python_ml_latest.sif 

# For train
python ./4_ml.py data.category=athlete data.data_dir=../../../data/clean/athlete data.batch_size.train=4 data.batch_size.val=8 optimizer.learning_rate=1e-4
python ./4_ml.py data.category=athlete data.data_dir=../../../data/origin/athlete data.batch_size.train=4 data.batch_size.val=8 optimizer.learning_rate=1e-4
# For test
python ./4_ml.py data.category=aircraft data.data_dir=../../../data/origin/aircraft data.batch_size.train=128 data.batch_size.val=256 optimizer.learning_rate=5e-4 general.is_train=False +model.weight_file=0325-132028/Loss2.8200_epoch10.bin
python ./4_ml.py data.category=athlete data.data_dir=../../../data/origin/athlete data.batch_size.train=4 data.batch_size.val=8 optimizer.learning_rate=1e-4 general.is_train=False +model.weight_file=0325-131401/Loss1.7903_epoch10.bin
python ./4_ml.py data.category=bird data.data_dir=../../../data/origin/bird data.batch_size.train=32 data.batch_size.val=64 optimizer.learning_rate=1e-3 general.is_train=False +model.weight_file=0325-132030/Loss0.9974_epoch10.bin
python ./4_ml.py data.category=bread data.data_dir=../../../data/origin/bread data.batch_size.train=32 data.batch_size.val=64 optimizer.learning_rate=5e-4 general.is_train=False +model.weight_file=0325-132028/Loss2.4758_epoch9.bin
python ./4_ml.py data.category=car data.data_dir=../../../data/origin/car data.batch_size.train=32 data.batch_size.val=64 optimizer.learning_rate=1e-3 general.is_train=False +model.weight_file=0325-132029/Loss1.3950_epoch9.bin
python ./4_ml.py data.category=director data.data_dir=../../../data/origin/director data.batch_size.train=128 data.batch_size.val=256 optimizer.learning_rate=1e-3 general.is_train=False +model.weight_file=0325-132029/Loss3.5289_epoch10.bin
python ./4_ml.py data.category=dog data.data_dir=../../../data/origin/dog data.batch_size.train=32 data.batch_size.val=64 optimizer.learning_rate=5e-4 general.is_train=False +model.weight_file=0325-132028/Loss1.8206_epoch8.bin
python ./4_ml.py data.category=us_politician data.data_dir=../../../data/origin/us_politician data.batch_size.train=128 data.batch_size.val=256 optimizer.learning_rate=5e-4 general.is_train=False +model.weight_file=0325-132036/Loss5.5134_epoch3.bin

# GCS
singularity shell --pwd /$HOME/Instance_level_recognition/app/object_detection $HOME/Instance_level_recognition/app/ml/gc_cli_latest.sif
gsutil -m cp -r gs://entity_dogs/ ./

gcloud auth login
gcloud config set project $PROJECT_ID
