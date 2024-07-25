#!/bin/bash

# GPUあたりに割り当てるプロセス数とGPUの総数
procs_per_gpu=4
ngpus=4

# 環境変数OMPI_COMM_WORLD_RANKに基づいて、割り当てるGPUを計算
gpu_id=$(($OMPI_COMM_WORLD_RANK / $procs_per_gpu % $ngpus))

# 計算されたGPU IDに基づいてCUDA_VISIBLE_DEVICESを設定し、MPIプログラムを実行
exec env CUDA_VISIBLE_DEVICES=$gpu_id "$@"
