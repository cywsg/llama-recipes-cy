# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ACCELERATE_USE_FSDP=true

torchrun --nnodes 1 --nproc_per_node 8  llama_qlora_finetuning.py \
	 --enable_fsdp --low_cpu_fsdp --fsdp_cpu_offload --use_fast_kernels \
	 --quantization --peft_method lora \
	 --model_name  /home/chong-yaw.wee/work/models/Llama-2-7b-32K  \
	 --dist_checkpoint_root_folder model_checkpoints \
	 --dist_checkpoint_folder llama-2-7b-32K-qlora \
	 --dataset sum_dataset \
	 --pure_bf16  --num_epochs 1 --lr 0.000002 \
	 --weight_decay 0.01 --batch_size_training 1 --micro_batch_size 1 
