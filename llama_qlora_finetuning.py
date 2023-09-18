# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List, Union

import fire
import torch
import transformers
from datasets import load_dataset
import os.path as osp
from tqdm import tqdm

# Unused imports removed
from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    BitsAndBytesConfig
)
import torch.distributed as dist

# Unused imports removed
from utils.train_utils import (
    set_tokenizer_params,
    train,
    evaluation,
    freeze_transformer_layers,
    check_frozen_layers_peft_model,
    setup,
    setup_environ_flags,
    cleanup,
    clear_gpu_cache,
    get_parameter_dtypes,
    print_model_size,
    get_policies,
    merge_weights_save_model
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    create_bnb_config,
    generate_dataset_config,
)
from peft import get_peft_model, TaskType, prepare_model_for_kbit_training
import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.utils.data import random_split, Subset


def main(**kwargs):
    # Update the configuration for the training and sharding process
    for k, v in kwargs.items():
        print(f"{k}: {v}")
    print("--------")

    update_config((train_config, fsdp_config), **kwargs)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )

    dataset_config = generate_dataset_config(train_config, kwargs)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    print(f"--> Full Training Set Length = {len(dataset_train)}")

    if len(dataset_config.test_split) > 0:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
    else:
        # split the data set to training data and validation data
        dataset = dataset_train
        print(f"--> Training Set Length Before Splitting = {len(dataset)}")
#        indices = torch.arange(1000)
#        dataset = Subset(dataset_train, indices)
        train_size = int(0.9 * len(dataset))
        dataset_train, dataset_val = random_split(dataset, [train_size, len(dataset) - train_size])

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size
    # gradient_accumulation_steps = train_config.gradient_accumulation_steps
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

#     # Load the pre-trained model and setup its configuration
#     model = AutoModelForCausalLM.from_pretrained(
#         train_config.model_name,
#         load_in_8bit=True if train_config.quantization else None,
#         device_map="auto" if train_config.quantization else None,
#     )
    
    if train_config.quantization:
        bnb_config = create_bnb_config()

    # Load the pre-trained model and setup its configuration
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        
        # print(f"train_config.enable_fsdp: {train_config.enable_fsdp}")
        # print(f"train_config.low_cpu_fsdp: {train_config.low_cpu_fsdp}")

        if rank == 0:
            if "yarn" in train_config.model_name.lower():
                from scaled_rope.configuration_llama import LlamaConfig
                from scaled_rope.modeling_llama_together_yarn import LlamaForCausalLM
                yarn_factor = 16.0
                yarn_config = LlamaConfig.from_pretrained(train_config.model_name)
                yarn_config.rope_scaling = {
                    "type": "yarn",
                    "factor": yarn_factor,
                    "original_max_position_embeddings": 4096
                }
                yarn_config.max_position_embeddings = int(yarn_factor * 4096)

                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    torch_dtype=torch.bfloat16,
                    config=yarn_config,
                    use_safetensors=False,
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config if train_config.quantization else None,
                    # load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_safetensors=False,
                    trust_remote_code=True,
                )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        print(f"train_config.enable_fsdp: {train_config.enable_fsdp}")
        print(f"train_config.low_cpu_fsdp: {train_config.low_cpu_fsdp}")
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            # load_in_8bit=True if train_config.quantization else None,
            quantization_config=bnb_config if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_safetensors=False,
        ) 

    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")    
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)
        
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        # print(f"{peft_config}")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
   
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")


    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )
        
    # Create DataLoaders for the training and validation dataset
    # print(dataset_train[0])
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
        
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader, 
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

    # save merge model
    if train_config.output_merged_dir:
        merge_weights_save_model(
            train_config.output_dir,
            train_config.output_merged_dir,
            tokenizer,
        )

if __name__ == "__main__":
    fire.Fire(main)
