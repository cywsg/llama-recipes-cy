#!/bin/bash

#SBATCH --job-name=Llama2-70b-finetuning-3nodes

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gpus-per-task=8     
#SBATCH --gres=gpu:8
#SBATCH --partition=defq
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Enable for A100
export FI_PROVIDER="efa"

echo NODELIST: $nodes_array
echo nodes: $nodes
echo head_node: $head_node
echo Node IP: $head_node_ip

export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME="ens"
export NCCL_SOCKET_IFNAME="enp"
#export FI_EFA_USE_DEVICE_RDMA=1

export NCCL_IB_SL=1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MELLANOX_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=0

# export NCCL_IB_TIMEOUT=22

# use fsdp
# export ACCELERATE_USE_FSDP=true

# srun  torchrun --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 llama_finetuning.py  --enable_fsdp --use_peft --peft_method lora

# srun  torchrun --nnodes 3 --nproc_per_node 8  llama_finetuning.py --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \

# srun  accelerate launch --multi_gpu --mixed_precision fp16 --num_processes 8 --num_machines 1 llama_finetuning.py --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
#         --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels  \
#         --model_name  /home/chong-yaw.wee/work/models/Llama-2-7b-hf  \
#         --dist_checkpoint_root_folder model_checkpoints \
#         --dist_checkpoint_folder sso-finetuned-4k  --pure_bf16 \
#         --dataset ssoj_dataset --num_epochs 1 --lr 0.000002 \
#         --weight_decay 0.01   --batch_size_training 2 --micro_batch_size 2

# srun  torchrun  --nnodes=1:3 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \

srun   --mpi=pmix -N 3 --ntasks-per-node=1 --gpus-per-task=8  \
	torchrun --nnodes=3 --nproc_per_node 8  --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
	llama_finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels  \
	--model_name  /home/chong-yaw.wee/work/models/Llama-2-70b-chat-hf  \
	--dist_checkpoint_root_folder model_checkpoints \
	--dist_checkpoint_folder sso-finetuned-4k  --pure_bf16 \
	--dataset ssoj_dataset --num_epochs 1 --lr 0.000002 \
	--weight_decay 0.01   --batch_size_training 4 --micro_batch_size 4
