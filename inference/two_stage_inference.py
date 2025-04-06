from transformers import T5Tokenizer, T5ForConditionalGeneration
#this page will be used to create inference which will be integrated with a frontend UI
def two_stage_inference(premise, hypothesis):
    # 1) Load Explainer checkpoint
    explainer_dir = "checkpoints/explainer"
    explainer_tokenizer = T5Tokenizer.from_pretrained(explainer_dir)
    explainer_model = T5ForConditionalGeneration.from_pretrained(explainer_dir)

    # 2) Generate Explanation
    prompt_expl = f"explain premise: {premise} hypothesis: {hypothesis}"
    inputs_expl = explainer_tokenizer(prompt_expl, return_tensors="pt")
    expl_outputs = explainer_model.generate(**inputs_expl, max_length=64, num_beams=2)
    explanation = explainer_tokenizer.decode(expl_outputs[0], skip_special_tokens=True)

    # 3) Load Predictor checkpoint
    predictor_dir = "checkpoints/predictor"
    predictor_tokenizer = T5Tokenizer.from_pretrained(predictor_dir)
    predictor_model = T5ForConditionalGeneration.from_pretrained(predictor_dir)

    # 4) Predict label from premise + hypothesis + explanation
    prompt_pred = f"predict premise: {premise} hypothesis: {hypothesis} explanation: {explanation}"
    inputs_pred = predictor_tokenizer(prompt_pred, return_tensors="pt")
    label_outputs = predictor_model.generate(**inputs_pred, max_length=5)
    label = predictor_tokenizer.decode(label_outputs[0], skip_special_tokens=True).strip()

    return explanation, label

if __name__ == "__main__":
    premise = "A man is riding a horse."
    hypothesis = "A person is on an animal."
    explanation, label = two_stage_inference(premise, hypothesis)

    print("Premise   :", premise)
    print("Hypothesis:", hypothesis)
    print("Explanation:", explanation)
    print("Label      :", label)
