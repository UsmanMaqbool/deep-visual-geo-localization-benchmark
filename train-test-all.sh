#!/bin/sh

PYTHON=${PYTHON:-"python3"}

FILES="/home/leo/usman_ws/models/graphvlad/benchmarking_vg/pitts30k-vgg16-triplet-lr0.001-26-Oct/*.pth"
export DATASETS_FOLDER=datasets_vg/datasets

echo "==========Testing============="
echo "=============================="

for RESUME in $FILES
do
  echo "Processing $RESUME ..."
  # take action on each file. $f store current file name
  python3 eval.py --dataset_name=pitts30k --backbone=vgg16 --aggregation=netvlad --pca_dim=4096 --resume=$RESUME  --pca_dataset_folder=pitts30k/images/train
  echo "============Done================"
  
done
