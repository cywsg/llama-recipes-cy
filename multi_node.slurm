#!/bin/bash

#SBATCH --job-name=Llama2-7b-finetuning-1node

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
### SBATCH --partition=train 
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Enable for A100
export FI_PROVIDER="efa"

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="ens"
export FI_EFA_USE_DEVICE_RDMA=1

# srun  torchrun --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 llama_finetuning.py  --enable_fsdp --use_peft --peft_method lora

# srun torchrun --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 llama_finetuning.py --enable_fsdp  \
#	 --model_name  /home/chong-yaw.wee/work/models/Llama-2-70b-hf  \
#	 --dist_checkpoint_root_folder model_checkpoints \
#	 --dist_checkpoint_folder sso-finetuned-4k  --pure_bf16 --use_fast_kernels \
#	 --dataset ssoj_dataset --num_epochs 1 --lr 0.000002 \
#	 --weight_decay 0.01   --batch_size_training 1 --micro_batch_size 1


srun  torchrun  --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
	 llama_finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels  \
	 --model_name  /home/chong-yaw.wee/work/models/Llama-2-7b-hf  \
	 --dist_checkpoint_root_folder model_checkpoints \
	 --dist_checkpoint_folder sso-finetuned-4k  --pure_bf16 \
	 --dataset ssoj_dataset --num_epochs 1 --lr 0.000002 \
	 --weight_decay 0.01   --batch_size_training 2 --micro_batch_size 2
