# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
from functools import partial
from .utils import Concatenator


def get_preprocessed_sum(dataset_config, tokenizer, split, max_length=4096, seed=42):
    # dataset = datasets.load_dataset("sum", split=split)
    dataset = datasets.load_dataset(
        "json",
        data_files="ft_datasets/summ_data_32k/judiciary_32k_train.jsonl",
        split="train"
    )
    print(f"-->Full Training Set Length = {len(dataset)}")

    prompt = (
        f"Summarize the following legal judgment:\n{{judgment}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def create_prompt_formats(sample):
        return {
            "text": prompt.format(
                judgment=sample["judgment"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    def preprocess_batch(batch, tokenizer, max_length):
        """
        Tokenizing a batch
        """
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )


    # dataset = dataset.map(create_prompt_formats)
    dataset = dataset.map(create_prompt_formats, remove_columns=list(dataset.features))
    print(f"-->Full Training Set Length (After Prompting) = {len(dataset)}")
    # print(dataset[0])

    # dataset = dataset.map(
    #     lambda sample: tokenizer(sample["text"]),
    #     batched=True,
    #     remove_columns=list(dataset.features),
    # ).map(Concatenator(chunk_size=max_length), batched=True)

    remove_columns = list(dataset.features)
#     dataset = dataset.map(
#         lambda sample: tokenizer(sample["text"], max_length=max_length,
#         truncation=True),
#         batched=True,
#         remove_columns=remove_columns,
#     )

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=remove_columns,
    ).map(Concatenator(chunk_size=max_length), batched=True)

    # Filter out samples that have input_ids exceeding max_length
    # dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    print(f"-->Full Training Set Length (After Tokenization) = {len(dataset)}")
    return dataset
    
