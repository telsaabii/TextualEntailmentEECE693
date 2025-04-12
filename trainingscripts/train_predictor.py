#!/usr/bin/env python3

"""
train_predictor.py
Train the T5 label predictor model (Stage 2) with YAML config.

Usage example:
  python train_predictor.py --config configs/base_predictor.yaml
"""

import os
import sys
import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.predictor_dataset import PredictorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictorTrainer")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_predictor.yaml",
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

    # Create output directories
    output_dir = config.get("output_dir", "checkpoints/predictor")
    os.makedirs(output_dir, exist_ok=True)
    best_model_dir = os.path.join(output_dir, "best")
    final_model_dir = os.path.join(output_dir, "final")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    device = config["device"]
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(config["model_name"]).to(device)

    # Optional: integrate wandb
    if config.get("use_wandb", False):
        import wandb
        wandb.init(project=config.get("wandb_project", "predictor-nli"))
        wandb.config.update(config)

    # Load data
    train_data = PredictorDataset(
        csv_path=config["data/processed/predictor_train.csv"],
        tokenizer_name=config["model_name"],
        max_len=config["max_len"],
        target_max_len=config["target_max_len"]
    )
    dev_data = PredictorDataset(
        csv_path=config["data/processed/predictor_dev.csv"],
        tokenizer_name=config["model_name"],
        max_len=config["max_len"],
        target_max_len=config["target_max_len"]
    )

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=config["batch_size"], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # For tracking best model
    best_accuracy = 0.0
    best_step = 0

    global_step = 0
    for epoch in range(config["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
        model.train()
        
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if config.get("use_wandb", False):
                import wandb
                wandb.log({"train_loss": loss.item(), "step": global_step})

            if global_step % config["eval_steps"] == 0:
                logger.info(f"Step {global_step} | Train Loss: {loss.item():.4f}")
                current_accuracy = evaluate(model, tokenizer, dev_loader, device)
                
                # Save best model
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_step = global_step
                    logger.info(f"New best model at step {global_step} with accuracy {best_accuracy:.4f}")
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)

    logger.info(f"Training complete. Best model was at step {best_step} with accuracy {best_accuracy:.4f}")

    # Final evaluation on dev set
    logger.info("Final evaluation on dev set after training completes.")
    final_accuracy = evaluate(model, tokenizer, dev_loader, device)
    
    # Save final model
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final model saved to {final_model_dir} with accuracy {final_accuracy:.4f}")
    
    # If the best model is better than the final model, load and return it
    if best_accuracy > final_accuracy:
        logger.info(f"Loading best model from step {best_step} (accuracy: {best_accuracy:.4f}) instead of final model (accuracy: {final_accuracy:.4f})")
        model = T5ForConditionalGeneration.from_pretrained(best_model_dir).to(device)
    
    # Copy best model to main output directory for easy access
    logger.info(f"Copying best model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Predictor model training complete. Best model saved to {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    main()