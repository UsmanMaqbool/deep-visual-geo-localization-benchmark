#!/bin/sh

PYTHON=${PYTHON:-"python3"}
DATE=$(date '+%d-%b') 
LOSS=$1
LR=0.001
ARCH=vgg16
DATASET=pitts30k
batchsize=1

if [ $# -ne 1 ]
  then
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    exit 1
fi

FILES="/home/leo/usman_ws/models/graphvlad/benchmarking_vg/${DATASET}-${ARCH}-${LOSS}-lr${LR}-${DATE}"
echo ${FILES}
export DATASETS_FOLDER=datasets_vg/datasets
python3 train.py --dataset_name=${DATASET} --lr=${LR} --backbone=${ARCH} --criterion=${LOSS} --pretrain=offtheshelf --train_batch_size=${batchsize} --save_dir=${FILES}

echo "==========Testing============="
FILES="${FILES}/*.pth"
echo ${FILES}

for RESUME in $FILES
do
  echo "Processing $RESUME ..."
  # take action on each file. $f store current file name
  python3 eval.py --dataset_name=${DATASET} --backbone=vgg16 --aggregation=netvlad --pca_dim=4096 --resume=$RESUME  --pca_dataset_folder=pitts30k/images/train
  echo "============Done================"
  
done

