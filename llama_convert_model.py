#!/usr/bin/env python
# coding=utf-8
"""
save pretrain model from pt model 
"""

import os
import sys
import argparse
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)

import torch


def parse_args():
    """
    This is parse the command line argument.
    :return: argument object
    """
    parser = argparse.ArgumentParser(description='save_pretrain_model.py')
    parser.add_argument('-base_model_path', type=str, default='/raid2/model/llama2_hf/Llama-2-7b-hf', help='base model path')
    parser.add_argument('-finetune_dir', type=str, default='model_checkpoints/fine-tuned-/raid2/model/llama2_hf/Llama-2-7b-hf', help='finetune checkpoints dir')
    parser.add_argument('-pt', type=str, default='Llama-2-7b-hf-0.pt', help='the pt model file')
    parser.add_argument('-save_output', type=str, default='save_model_llama2',  help='save pre_trained model path')

    args = parser.parse_args()
    return args



args_c = None

if __name__ == "__main__":

    args_c = parse_args()

    pt_model_path = args_c.finetune_dir + '/' + args_c.pt

    if not os.path.isfile(pt_model_path):
        sys.stderr.write("File path {} does not exist. Exiting...\n".format(pt_model_path))
        sys.exit()

    # Load model
    print("start load base model")
    data_type = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args_c.base_model_path, device_map="auto", low_cpu_mem_usage=True, use_safetensors=False, torch_dtype=data_type)

    print("start load finetuned pt  model")
    model.load_state_dict(torch.load(pt_model_path))

    print("start save pretrained model")
    model.save_pretrained(args_c.save_output, safe_serialization=False)

    tokenizer = AutoTokenizer.from_pretrained(args_c.base_model_path)
    tokenizer.save_pretrained(args_c.save_output, legacy_format=False)



