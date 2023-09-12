# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "ft_datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "ft_datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_data.json"


@dataclass
class ssoj_dataset:
    dataset: str = "ssoj_dataset"
    train_split: str = "train"
    test_split: str = ""
    data_path: str = "ft_datasets/finetuning_sso_j.txt"
    input_length: int = 4096


@dataclass
class ssoj_short_dataset:
    dataset: str = "ssoj_short_dataset"
    train_split: str = "train"
    test_split: str = ""
    data_path: str = "ft_datasets/finetuning_sso_j_short.txt"
    input_length: int = 2048


@dataclass
class wiki_dataset:
    dataset: str = "wiki_dataset"
    train_split: str = "train"
    test_split: str = ""
    data_path: str = "ft_datasets/wiki_demo.txt"
    input_length: int = 2048


@dataclass
class sum_dataset:
    dataset: str =  "sum_dataset"
    train_split: str = "train"
    test_split: str = ""
    data_path: str = "ft_datasets/summ_data_32k/judiciary_32k_train.jsonl"
    input_length: int = 32768
