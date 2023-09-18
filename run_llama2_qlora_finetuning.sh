export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export ACCELERATE_USE_FSDP=true

# torchrun --nnodes 1 --nproc_per_node 1 llama_qlora_finetuning.py \
# 	  --use_peft \
#	  --peft_method lora \
#	  --pure_bf16 \
#         --num_epochs 20 \
#         --lr 0.00002 \
#         --weight_decay 0.05 \
#         --batch_size_training 4 \
#         --micro_batch_size 4 \
#         --val_batch_size 1 \
#	  --dataset sum_dataset \
# 	 --model_name /home/chong-yaw.wee/work/models/Llama-2-7b-32K \
# 	 --output_dir model_checkpoints/Llama-2-7b-32K-lora \
# 	 --output_merged_dir model_checkpoints/Llama-2-7b-32K-merged \
#	 --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels \
	 # --dist_checkpoint_root_folder model_checkpoints \
         # --dist_checkpoint_folder llama-2-7b-32K-lora \
	 # --gradient_accumulation_steps 10 
         # --quantization


# torchrun --nnodes 1 --nproc_per_node 1 llama_qlora_finetuning.py \
#         --use_peft \
#         --peft_method lora \
#         --pure_bf16 \
#         --num_epochs 20 \
#         --lr 0.00002 \
#         --weight_decay 0.05 \
#         --batch_size_training 4 \
#         --micro_batch_size 4 \
#         --val_batch_size 1 \
#         --dataset sum_dataset \
#         --model_name /home/chong-yaw.wee/work/models/Llama-2-7B-32K-Instruct \
#         --output_dir model_checkpoints/Llama-2-7B-32K-Instruct-lora \
#         --output_merged_dir model_checkpoints/Llama-2-7B-32K-Instruct-merged \
#         --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels


torchrun --nnodes 1 --nproc_per_node 1 llama_qlora_finetuning.py \
         --use_peft \
         --peft_method lora \
         --pure_bf16 \
         --num_epochs 20 \
         --lr 0.00002 \
         --weight_decay 0.05 \
         --batch_size_training 4 \
         --micro_batch_size 4 \
         --val_batch_size 1 \
         --dataset sum_dataset \
         --model_name /home/chong-yaw.wee/work/models/Yarn-Llama-2-13b-64k \
         --output_dir model_checkpoints/Yarn-Llama-2-13b-64k-lora \
         --output_merged_dir model_checkpoints/Yarn-Llama-2-13b-64k-merged \
         --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels \



# torchrun --nnodes 1 --nproc_per_node 1 llama_qlora_finetuning.py \
#          --use_peft \
#          --peft_method lora \
#          --pure_bf16 \
#          --num_epochs 20 \
#          --lr 0.00002 \
#          --weight_decay 0.05 \
#          --batch_size_training 4 \
#          --micro_batch_size 4 \
#          --val_batch_size 1 \
#          --dataset sum_dataset \
#          --model_name /home/chong-yaw.wee/work/models/Yarn-Llama-2-13b-128k \
#          --output_dir model_checkpoints/Yarn-Llama-2-13b-128k-lora \
#          --output_merged_dir model_checkpoints/Yarn-Llama-2-13b-128k-merged \
#          --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels \
