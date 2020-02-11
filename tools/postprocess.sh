#!/usr/bin/env bash

result='/data1/wzz/down4/'

python prepare_folder.py --result=$result

testpath='/data1/wzz/dtu/'
testlist='/data1/wzz/test.txt'
use_mmp=True

downsample4_dir=$result'/0'
downsample8_dir=$result'/1'
downsample16_dir=$result'/2'

python filter_fusion.py --testpath=$testpath --testlist=$testlist --gt_folder=$gt_folder --use_mmp=$use_mmp --outdir_down4=$downsample4_dir --outdir_down8=$downsample8_dir --outdir_down16=$downsample16_dir 
