#!/bin/bash
source env.sh
data=$(date +"%m%d")
n=4
batch=4
d=192
interval_scale=1.06
lr=0.001
lr_scheduler=cosinedecay
loss=mvsnet_loss_l1norm
#optimizer=RAdam
optimizer=Adam
fea_net=UNetDS2GN
cost_net=RegNetUS0GN
loss_w=4
inverse_depth=True
syncbn=False
origin_size=False
cost_aggregation=91
view_num=3
name=${data}_g${n}_b${batch}_d${d}_is${interval_scale}_${loss}_lr${lr}_op${optimizer}_${lrepochs}_sh${lr_scheduler}_${fea_net}_${cost_net}_id${inverse_depth}_os${origin_size}_sbn${syncbn}_cg${cost_aggregation}_vn${view_num}_lw${loss_w}
now=$(date +"%Y%m%d_%H%M%S")
echo $name
echo $now
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
        --dataset=dtu_yao \
        --batch_size=${batch} \
        --trainpath=$MVS_TRAINING \
        --loss=${loss} \
        --lr=${lr} \
        --loss_w=$loss_w \
        --lr_scheduler=$lr_scheduler \
        --optimizer=$optimizer \
        --view_num=$view_num \
        --inverse_depth=${inverse_depth} \
        --origin_size=$origin_size \
        --syncbn=$syncbn \
        --cost_aggregation=$cost_aggregation \
        --ngpu=${n} \
        --fea_net=${fea_net} \
        --cost_net=${cost_net} \
        --trainlist=lists/dtu/train.txt \
        --vallist=lists/dtu/val.txt \
        --testlist=lists/dtu/test.txt \
        --numdepth=$d \
        --interval_scale=$interval_scale \
        --logdir=./checkpoints_gn/${name} &
        2>&1|tee ./logs/${name}-${now}.log &
