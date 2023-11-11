#!/bin/bash
# Acceptable values include `125m`, `350m`, `1.3b`, `2_7b`, `6_7b`, `13b`, `30b`, `66b`, `175b`
export MODEL="350m"
export BS="1"
#export GPUNUM="1"
export DATE=$(date '+%Y_%m_%d_%H_%M_%S')

mkdir -p ./logs

#LD_PRELOAD=/root/MLInsight/build/Runtime/libTrace-Auto-CUDA.so 
#deepspeed train_ds_opt.py \
LD_PRELOAD=`realpath ../../build/Runtime/libmlinsight.so` deepspeed train_ds_opt.py \
    --pretrain \
    --model_name_or_path facebook/opt-${MODEL} \
    --config_name ${MODEL} \
    --ds_config_name stage3-offload \
    --batch_size ${BS}
