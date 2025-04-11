#!/usr/bin/env python3

"""
train_vanilla.py
Train a T5 model to predict NLI labels (entailment, contradiction, neutral) 
from premise and hypothesis pairs.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLIPredictor")

class NLIPredictorDataset(Dataset):
    """Dataset for NLI prediction task using T5."""
    
    def __init__(self, csv_path, tokenizer_name, max_len=128, target_max_len=10):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.target_max_len = target_max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        premise = row['premise']
        hypothesis = row['hypothesis']
        label = row['label']
        
        # Format input text
        input_text = f"premise: {premise} hypothesis: {hypothesis}"
        
        # Tokenize inputs and labels
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer.encode_plus(
            label,
            None,
            add_special_tokens=True,
            max_length=self.target_max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nli_predictor.yaml",
                        help="Path to the YAML configuration file.")
    return parser.parse_args()

def load_config(config_path):
    """Load YAML configuration into a Python dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def evaluate(model, tokenizer, dev_loader, device):
    """Runs a basic evaluation loop (accuracy, classification report) on the dev set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=5)
            preds = [tokenizer.decode(g, skip_special_tokens=True).strip().lower() for g in outputs]
            targets = [tokenizer.decode(t, skip_special_tokens=True).strip().lower() for t in labels]

            all_preds.extend(preds)
            all_labels.extend(targets)

    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Dev Accuracy: {accuracy:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, zero_division=0))
    model.train()
    return accuracy

def main():
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}: {config}")

    # Optional: integrate wandb
    if config.get("use_wandb", False):
        import wandb
        wandb.init(project=config.get("wandb_project", "nli-predictor"))
        wandb.config.update(config)

    device = config["device"]
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(config["model_name"]).to(device)

    # Load data
    train_data = NLIPredictorDataset(
        csv_path=config["train_csv"],
        tokenizer_name=config["model_name"],
        max_len=config["max_len"],
        target_max_len=config["target_max_len"]
    )
    dev_data = NLIPredictorDataset(
        csv_path=config["dev_csv"],
        tokenizer_name=config["model_name"],
        max_len=config["max_len"],
        target_max_len=config["target_max_len"]
    )

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=config["batch_size"], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    best_accuracy = 0.0
    global_step = 0
    for epoch in range(config["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1

            if config.get("use_wandb", False):
                import wandb
                wandb.log({"train_loss": loss.item(), "step": global_step})

            if global_step % config["eval_steps"] == 0:
                logger.info(f"Step {global_step} | Train Loss: {loss.item():.4f}")
                accuracy = evaluate(model, tokenizer, dev_loader, device)
                
                # Save model if it's the best so far
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    logger.info(f"New best accuracy: {best_accuracy:.4f} - Saving model")
                    model.save_pretrained(config["output_dir"])
                    tokenizer.save_pretrained(config["output_dir"])
                
        # Log average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    logger.info("Final evaluation on dev set after training completes.")
    evaluate(model, tokenizer, dev_loader, device)

    # Save final model
    final_output_dir = os.path.join(config["output_dir"], "final")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"Final NLI predictor model saved to {final_output_dir}")

if __name__ == "__main__":
    main()