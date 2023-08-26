import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    # read training arguments from command arguments
    data_type = torch.float16
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model_dir = "/home/chong-yaw.wee/work/models/Llama-2-7b-chat-hf"

    # download the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True, device_map="auto", load_in_8bit=True, low_cpu_mem_usage=True, torch_dtype=data_type) # base model

    # save model
    model.save_pretrained(model_dir, safe_serialization=False)

    # save tokenizer
    tokenizer.save_pretrained(model_dir, legacy_format=False)

    print(f'Save model and tokenizer to {model_dir}!')
