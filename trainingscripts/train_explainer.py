#!/usr/bin/env python3

"""
train_explainer.py
Train the T5 explainer model using a YAML config file.

Usage example:
  python train_explainer.py --config configs/base_explainer.yaml
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

# Make sure we can import from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.explainer_dataset import ExplainerDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExplainerTrainer")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_explainer.yaml",
                        help="Path to the YAML configuration file.")
    return parser.parse_args()

def load_config(config_path):
    """Load YAML configuration into a Python dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def evaluate(model, tokenizer, dev_loader, device):
    """
    Runs a basic evaluation loop on the dev set, logging predictions vs. gold labels.
    Returns an approximate score (match fraction) for model selection.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    exact_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=2
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            targets = [tokenizer.decode(t, skip_special_tokens=True) for t in labels]

            all_preds.extend(preds)
            all_targets.extend(targets)
            
            # Count exact matches for a simple metric
            for pred, target in zip(preds, targets):
                if pred == target:
                    exact_matches += 1
                total_samples += 1

    logger.info(f"Evaluated {len(all_preds)} samples on dev set.")
    for i in range(min(2, len(all_preds))):
        logger.info(f"[PRED] {all_preds[i]}\n[TARG] {all_targets[i]}")
    
    # Calculate a simple match score for model selection
    match_score = exact_matches / total_samples if total_samples > 0 else 0
    logger.info(f"Exact match score: {match_score:.4f} ({exact_matches}/{total_samples})")

    model.train()
    return match_score

def main():
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}: {config}")
    
    # Create output directory if it doesn't exist
    output_dir = config.get("output_dir", "checkpoints/explainer")
    os.makedirs(output_dir, exist_ok=True)
    best_model_dir = os.path.join(output_dir, "best")
    final_model_dir = os.path.join(output_dir, "final")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    device = config["device"]  # e.g., "cuda" or "cpu"
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(config["model_name"]).to(device)

    # Optional: integrate wandb if needed
    if config.get("use_wandb", False):
        import wandb
        wandb.init(project=config.get("wandb_project", "explainable-nli"))
        wandb.config.update(config)

    # Load data
    train_data = ExplainerDataset(
        csv_path=config["data/processed/explainer_train.csv"],
        tokenizer_name=config["model_name"],
        max_len=config["max_len"],
        target_max_len=config["target_max_len"]
    )
    dev_data = ExplainerDataset(
        csv_path=config["data/processed/explainer_dev.csv"],
        tokenizer_name=config["model_name"],
        max_len=config["max_len"],
        target_max_len=config["target_max_len"]
    )

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=config["batch_size"], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    
    # For tracking best model
    best_score = -1
    best_step = 0

    global_step = 0
    for epoch in range(config["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
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
                current_score = evaluate(model, tokenizer, dev_loader, device)
                
                # Save best model
                if current_score > best_score:
                    best_score = current_score
                    best_step = global_step
                    logger.info(f"New best model at step {global_step} with score {best_score:.4f}")
                    model.save_pretrained(best_model_dir)
                    tokenizer.save_pretrained(best_model_dir)

    logger.info(f"Training complete. Best model was at step {best_step} with score {best_score:.4f}")
    
    # Final evaluation on dev set
    logger.info("Final evaluation on dev set...")
    final_score = evaluate(model, tokenizer, dev_loader, device)
    
    # Save final model
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Final model saved to {final_model_dir} with score {final_score:.4f}")
    
    # If the best model is better than the final model, load and return it
    if best_score > final_score:
        logger.info(f"Loading best model from step {best_step} (score: {best_score:.4f}) instead of final model (score: {final_score:.4f})")
        model = T5ForConditionalGeneration.from_pretrained(best_model_dir).to(device)
        
    # Copy best model to main output directory for easy access
    logger.info(f"Copying best model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Explainer model training complete. Best model saved to {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    main()