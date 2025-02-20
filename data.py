import pandas as pd
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import torch

def load_custom_dataset():
    ds = hf_load_dataset("DSWF/medical_chatbot")
    # Load the dataset using Pandas
    df = ds['train'].to_pandas()
    # Print the column names for debugging
    print("Columns in the dataset:", df.columns)
    # Ensure the DataFrame has a column "Answer" with the text data
    if "Answer" not in df.columns:
        raise ValueError("The dataset does not contain a 'Answer' column.")
    dataset = Dataset.from_pandas(df)
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=50):
    def tokenize_function(example):
        return tokenizer(example["Answer"], truncation=True, max_length=max_length)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def split_dataset(tokenized_dataset, test_size=0.1):
    return tokenized_dataset.train_test_split(test_size=test_size)

def collate_fn(batch, pad_token_id):
    # Each batch element is a dict with an 'input_ids' key.
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    return padded_input_ids

def get_dataloaders(tokenized_dataset, tokenizer, batch_size=16):
    split_ds = split_dataset(tokenized_dataset, test_size=0.1)
    train_dataset = split_ds["train"]
    val_dataset = split_ds["test"]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = load_custom_dataset()
    tokenized_ds = tokenize_dataset(ds, tokenizer, max_length=50)
    train_loader, val_loader = get_dataloaders(tokenized_ds, tokenizer, batch_size=16)
    for batch in train_loader:
        print(batch.shape)
        break