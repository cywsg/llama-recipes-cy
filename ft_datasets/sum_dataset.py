opyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
from .utils import Concatenator


def get_preprocessed_sum(dataset_config, tokenizer, split):
    # dataset = datasets.load_dataset("sum", split=split)
    dataset = datasets.load_dataset(
        "json",
        data_files="ft_datasets/summ_data_32k/judiciary_32k_train.jsonl",
        split="train"
    )

    prompt = (
        f"Summarize the following legal judgement:\n{{judgement}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def create_prompt_formats(sample):
        return {
            "text": prompt.format(
                judgement=sample["judgement"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    # dataset = dataset.map(create_prompt_formats)
    dataset = dataset.map(create_prompt_formats, remove_columns=list(dataset.features))

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
    
