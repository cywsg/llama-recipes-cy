
# python llama_convert_model.py \
#	-base_model_path /dev/shm/models/Llama-2-70b-hf \
#	-finetune_dir /dev/shm/models/finetuned-4k/Llama-2-70b-hf \
#	-pt Llama-2-70b-hf-0.pt \
#	-save_output model_checkpoints/llama2-hf/Llama-2-70b-sso-hf


python llama_convert_model.py \
	-base_model_path /dev/shm/models/Llama-2-70b-chat-hf \
	-finetune_dir /dev/shm/models/finetuned-4k/Llama-2-70b-chat-hf \
	-pt Llama-2-70b-chat-hf-0.pt  \
	-save_output model_checkpoints/llama2-hf/Llama-2-70b-chat-sso-hf
