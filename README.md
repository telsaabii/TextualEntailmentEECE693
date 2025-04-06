# 📚 Two-Stage Textual Entailment with Explainability

This project implements **two-stage textual entailment** using **T5 models** on the **e-SNLI dataset** with further finetuning on a custom philosophy dataset. It is designed for research and educational use (EECE 693).

## 🔍 Project Goals

1. **Generate human-readable explanations** for why a premise entails, contradicts, or is neutral to a hypothesis.
2. **Use those explanations** to improve the accuracy and interpretability of final **entailment label predictions** (Stage 2).

---

## 📁 Folder Structure & Purpose

---

### `checkpoints/`
**Purpose:** Stores trained T5 model weights after training.

- `explainer/`: The Stage 1 T5 Explainer model
- `predictor/`: The Stage 2 T5 Predictor model

**Contents:**
- `pytorch_model.bin`, `config.json`, `tokenizer_config.json`, etc.

> ✅ **Future**: These checkpoints are used in inference or future fine-tuning.

---

### `configs/`
**Purpose:** Contains YAML config files for training and inference.

- `base_explainer.yaml`, `custom_explainer.yaml`: Hyperparams for Stage 1 training
- `base_predictor.yaml`, `custom_predictor.yaml`: Hyperparams for Stage 2 training
- `inference.yaml`: Inference model paths and generation settings (beam size, max length)

> ✅ **Future**: Add more configs for experiments with different learning rates, epochs, etc.

---

### `data/`
**Purpose:** Handles all data preparation and dataset logic.

#### `processed/`
- Preprocessed CSVs from e-SNLI:
  - `esnli_train_1.csv`, `esnli_train_2.csv`
  - `esnli_dev.csv`, `esnli_test.csv`

#### Key Scripts:
- `2StageDataPrep.py`: Prepares data for explainer/predictor stages
- `explainer_dataset.py`: PyTorch `Dataset` for generating explanations
- `predictor_dataset.py`: PyTorch `Dataset` for predicting labels using explanations

> ✅ **Future**: Extend for other domains (e.g., philosophy), or add data augmentation.

---

### `evaluation/`
**Purpose:** Contains scripts for evaluating explanation quality, model interpretability, and ablation studies.

- `explanation_eval.py`: Computes BLEU / ROUGE scores for explanations
- `attention_eval.py`: Visualizes attention weights from T5 (cross-attention heatmaps)
- `ablation_eval.py`: Compares label prediction accuracy **with vs. without** explanations

> ✅ **Future**: Add more evaluation metrics (faithfulness, plausibility, etc.).

---

### `inference/`
**Purpose:** End-to-end pipeline for inference.

- `two_stage_inference.py`: Runs both stages:
  1. Generates explanation using Explainer model
  2. Predicts label using Predictor model + generated explanation

> ✅ **Future**: Extend with single-stage or batched inference, ensemble experiments.

---

### `trainingscripts/`
**Purpose:** Contains training logic for both model stages.

- `train_explainer.py`: Trains the T5 Explainer model using `base_explainer.yaml`
- `train_predictor.py`: Trains the T5 Predictor model using `base_predictor.yaml`

> ✅ **Future**: Add variants for multitask learning, early stopping, larger models, etc.

---

### `WebApp.py`
**Purpose:** Placeholder for turning the system into a **web API** or **demo app**.

- 🧠 Idea: Load explainer/predictor models once, and serve predictions via a web route.
- 📦 Return explanation and label in a structured response.

> ✅ **Future**: Integrate with **FastAPI** or **Flask** for a working UI or API.

---

### `requirements.txt`
**Purpose:** Specifies all Python dependencies required to run the code.

---

## 🚀 Setup Instructions

```bash
# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt
```

---

## 🧪 Example Usage

### 1. Data Preparation
```bash
python data/2StageDataPrep.py
```

### 2. Train Explainer Model (Stage 1)
```bash
python trainingscripts/train_explainer.py --config configs/base_explainer.yaml
```

### 3. Train Predictor Model (Stage 2)
```bash
python trainingscripts/train_predictor.py --config configs/base_predictor.yaml
```

### 4. Run Inference
```bash
python inference/two_stage_inference.py \
  --config configs/inference.yaml \
  --premise "A man is riding a horse." \
  --hypothesis "A person is on an animal."
```

---

## 📌 `requirements.txt`

```txt
# Core
torch>=1.10
transformers>=4.25
tqdm
pandas
numpy
PyYAML
scikit-learn

# Evaluation
rouge-score
nltk
sacrebleu

# Visualization (optional)
matplotlib

# Logging (optional)
wandb

# Web serving (optional for WebApp.py)
fastapi
uvicorn
```

---

## 📓 Notes

- All model saving/loading uses the HuggingFace `transformers` interface via `.save_pretrained()` and `.from_pretrained()`.
- You can train on **GPU** or fallback to **CPU**.
- The explainer and predictor are completely decoupled and can be trained separately.

---

## 👥 Authors / Contributors

- 
