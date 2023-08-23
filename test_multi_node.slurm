#!/bin/sh

export NCCL_IB_SL=1 
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1 
export NCCL_DEBUG=INFO 
export NCCL_NET_GDR_LEVEL=0 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export MELLANOX_VISIBLE_DEVICES=0,1,6,7 

srun --mpi=pmix -N 2 --ntasks-per-node=1 --gpus-per-task=8 --container-image=nvcr.io#nvidia/pytorch:23.04-py3 -J mlnx-comms:mlperf -t 00:02:00 all_reduce_perf_mpi --minbytes 4 --maxbytes 4G -f2 -g8 -n1000
