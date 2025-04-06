import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

class ExplainerDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="t5-base", max_len=128, target_max_len=64):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.target_max_len = target_max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_text = f"explain premise: {row['premise']} hypothesis: {row['hypothesis']}"
        target_text = row['explanation']

        enc_in = self.tokenizer(
            input_text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        enc_tgt = self.tokenizer(
            target_text, max_length=self.target_max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids": enc_in.input_ids.squeeze(),
            "attention_mask": enc_in.attention_mask.squeeze(),
            "labels": enc_tgt.input_ids.squeeze(),
        }
