#!/bin/bash

python train/train.py \
                      --batch_size=10 \
                      --workers=50 \
                      --dataset_dir=/home/zxm/dataset/dataset/nyuv2/nyu_depth_v2 \
                      --save_dir=checkpoints \
                      --epoches=20 \
                      --lr=0.001 \
                      --lr_decay=0.2 \
                      --lr_decay_step=8 \
                      --test_rate=10 \
                      --log_rate=1
