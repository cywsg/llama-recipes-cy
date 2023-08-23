# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ACCELERATE_USE_FSDP=true

nohup torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp  \
	 --low_cpu_fsdp --fsdp_cpu_offload  \
	 --model_name  /home/chong-yaw.wee/work/models/Llama-2-70b-hf  \
	 --dist_checkpoint_root_folder model_checkpoints \
	 --dist_checkpoint_folder sso-finetuned-4k  --pure_bf16 --use_fast_kernels \
	 --dataset ssoj_dataset --num_epochs 1 --lr 0.000002 \
	 --weight_decay 0.01   --batch_size_training 1 --micro_batch_size 1 
