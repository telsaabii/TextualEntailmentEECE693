#this page will be used to create inference which will be integrated with a frontend UI
import yaml
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_config(path="configs/inference.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def two_stage_inference(premise, hypothesis, config):
    # Load explainer config
    explainer_dir = config["explainer_model_path"]
    expl_cfg = config["explainer"]
    explainer_tokenizer = T5Tokenizer.from_pretrained(explainer_dir)
    explainer_model = T5ForConditionalGeneration.from_pretrained(explainer_dir)

    # Generate Explanation
    prompt_expl = f"explain premise: {premise} hypothesis: {hypothesis}"
    inputs_expl = explainer_tokenizer(prompt_expl, return_tensors="pt")
    expl_outputs = explainer_model.generate(
        input_ids=inputs_expl["input_ids"],
        attention_mask=inputs_expl["attention_mask"],
        max_length=expl_cfg.get("max_length", 64),
        num_beams=expl_cfg.get("num_beams", 2),
        do_sample=expl_cfg.get("do_sample", False),
        temperature=expl_cfg.get("temperature", 1.0),
        top_k=expl_cfg.get("top_k", 50),
        top_p=expl_cfg.get("top_p", 0.95)
    )
    explanation = explainer_tokenizer.decode(expl_outputs[0], skip_special_tokens=True)

    # Load predictor config
    predictor_dir = config["predictor_model_path"]
    pred_cfg = config["predictor"]
    predictor_tokenizer = T5Tokenizer.from_pretrained(predictor_dir)
    predictor_model = T5ForConditionalGeneration.from_pretrained(predictor_dir)

    # Predict label
    prompt_pred = f"predict premise: {premise} hypothesis: {hypothesis} explanation: {explanation}"
    inputs_pred = predictor_tokenizer(prompt_pred, return_tensors="pt")
    label_outputs = predictor_model.generate(
        input_ids=inputs_pred["input_ids"],
        attention_mask=inputs_pred["attention_mask"],
        max_length=pred_cfg.get("max_length", 5),
        num_beams=pred_cfg.get("num_beams", 1),
        do_sample=pred_cfg.get("do_sample", False),
        temperature=pred_cfg.get("temperature", 1.0)
    )
    label = predictor_tokenizer.decode(label_outputs[0], skip_special_tokens=True).strip()

    return explanation, label

if __name__ == "__main__":
    premise = "A man is riding a horse."
    hypothesis = "A person is on an animal."

    config = load_config()
    explanation, label = two_stage_inference(premise, hypothesis, config)

    print("Premise   :", premise)
    print("Hypothesis:", hypothesis)
    print("Explanation:", explanation)
    print("Label      :", label)
