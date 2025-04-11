# 📚 Two-Stage Textual Entailment with Explainability

In this project, we adopt a two-stage approach to textual entailment by first generating human-readable explanations using a T5-based explainer model, and subsequently using those explanations as input to a second T5-based predictor model that classifies the entailment label. This architecture is inspired by the structure of the e-SNLI dataset, which uniquely includes natural language explanations alongside premise-hypothesis-label triples. While most NLI systems leverage only the final label, our approach captures the intermediate reasoning process that humans often rely on when making inferences. This modular setup enhances interpretability and allows for controlled experimentation—such as ablation studies comparing prediction accuracy with and without explanations. By separating the explanation and classification stages, we enable fine-grained analysis of model behavior, attention patterns, and explanation quality, making this architecture highly suitable for explainable AI research and transferable to more philosophical or ethical reasoning domains which we will also explore.

## 🔍 Project Goals

1. **Generate human-readable explanations** for why a premise entails, contradicts, or is neutral to a hypothesis (Stage 1).
2. **Use those explanations** to improve the accuracy and interpretability of final **entailment label predictions** (Stage 2).

Through this project, we are hoping to address the following:
1. How does the quality and nature of reasoning differ between a model trained solely on e-SNLI and one fine-tuned on a philosophy-specific entailment dataset?
2. Does explicitly generating natural language explanations (reasoning) improve the model’s performance on entailment classification tasks?
3. What attention patterns are associated with different entailment labels, and how do these patterns differ between everyday (e-SNLI) and philosophical reasoning?
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

## 📓 Notes

- All model saving/loading uses the HuggingFace `transformers` interface via `.save_pretrained()` and `.from_pretrained()`.
- You can train on **GPU** or fallback to **CPU**.
- The explainer and predictor are completely decoupled and can be trained separately.

---

## 👥 Authors / Contributors

- 
