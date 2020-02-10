#!/usr/bin/env bash
source env.sh
n=1
model=mvsnet
data=$(date +"%m%d")
batch=1
lr=0.001
lr_scheduler=cosinedecay
loss=mvsnet_loss_l1norm
fea_net=UNetDS2GN
cost_net=RegNetUS0GN
inverse_depth=True
syncbn=True
origin_size=False
cost_aggregation=91
refine=False
save_depth=True
fusion=True
checkpoint_list=(11)
ckpt=./checkpoints/UNetDS2GN_RegNetUS0GN/model_0000
now=$(date +"%Y%m%d_%H%M%S")
echo $now
for idx in ${checkpoint_list[@]}
do
name=testPoint_${data}_g${n}_b${batch}_${model}_inversecv_${fea_net}_${cost_net}_id${inverse_depth}_sb${syncbn}_os${origin_size}_ca${cost_aggregation}_rf${refine}_sd${save_depth}_fu${fusion}_${part}_ckpt${idx}
echo $name
echo 'process light'${light_idx}
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
        --dataset=dtu_yao_eval \
        --model=$model \
        --syncbn=$syncbn \
        --batch_size=${batch} \
        --inverse_cost_volume \
        --inverse_depth=${inverse_depth} \
        --origin_size=$origin_size \
        --syncbn=$syncbn \
        --refine=$refine \
        --save_depth=$save_depth \
        --fusion=False \
        --cost_aggregation=$cost_aggregation \
        --ngpu=${n} \
        --fea_net=${fea_net} \
        --cost_net=${cost_net} \
        --pyramid=0 \
        --testpath=$DTU_TESTING \
        --testlist=lists/dtu/test.txt \
        --loadckpt=$ckpt${idx}.ckpt \
        --outdir=./outputs_ms_1101 &
CUDA_VISIBLE_DEVICES=1 python -u eval.py \
        --dataset=dtu_yao_eval \
        --model=$model \
        --syncbn=$syncbn \
        --batch_size=${batch} \
        --inverse_cost_volume \
        --inverse_depth=${inverse_depth} \
        --origin_size=$origin_size \
        --syncbn=$syncbn \
        --refine=$refine \
        --save_depth=$save_depth \
        --fusion=False \
        --cost_aggregation=$cost_aggregation \
        --ngpu=${n} \
        --fea_net=${fea_net} \
        --cost_net=${cost_net} \
        --pyramid=1 \
        --testpath=$DTU_TESTING \
        --testlist=lists/dtu/test.txt \
        --loadckpt=$ckpt${idx}.ckpt \
        --outdir=./outputs_ms_1101 &
CUDA_VISIBLE_DEVICES=2 python -u eval.py \
        --dataset=dtu_yao_eval \
        --model=$model \
        --syncbn=$syncbn \
        --batch_size=${batch} \
        --inverse_cost_volume \
        --inverse_depth=${inverse_depth} \
        --origin_size=$origin_size \
        --syncbn=$syncbn \
        --refine=$refine \
        --save_depth=True \
        --fusion=False \
        --cost_aggregation=$cost_aggregation \
        --ngpu=${n} \
        --fea_net=${fea_net} \
        --cost_net=${cost_net} \
        --pyramid=2 \
        --testpath=$DTU_TESTING \
        --testlist=lists/dtu/test.txt \
        --loadckpt=$ckpt${idx}.ckpt \
        --outdir=./outputs_ms_1101 &
done
