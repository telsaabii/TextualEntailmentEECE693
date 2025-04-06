#to be fixed
"""
explanation_eval.py
Compute BLEU and ROUGE scores for generated explanations vs. gold.
Use this script after training the explainer model, or at eval steps.
"""

'''
USAGE:
Generate a predictions.txt and references.txt in your train_explainer.py or separate inference script.

Then:
Terminal
python evaluation/explanation_eval.py --pred_file predictions.txt --gold_file references.txt
It will print BLEU, ROUGE-1, ROUGE-2, ROUGE-L.
'''

'''
Generating prediction.txt and references.txt:
in train_explainer.py evaluate: 
# After collecting `all_preds` and `all_targets` in evaluate()
with open("predictions.txt", "w") as fp:
    for p in all_preds:
        fp.write(p + "\n")

with open("references.txt", "w") as fp:
    for t in all_targets:
        fp.write(t + "\n")

'''

import argparse
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_bleu(hypotheses, references):
    """
    hypotheses: list of strings (model predictions)
    references: list of strings (gold explanations)
    returns float BLEU score
    """
    # Tokenize each sentence
    smoothie = SmoothingFunction().method1
    ref_tokenized = [[r.split()] for r in references]  # list of list of list
    hyp_tokenized = [h.split() for h in hypotheses]

    # nltk corpus_bleu expects references in the form [[ref1], [ref2], ...]
    bleu = corpus_bleu(ref_tokenized, hyp_tokenized, smoothing_function=smoothie)
    return bleu

def compute_rouge(hypotheses, references):
    """
    Use `rouge_score` library to get average ROUGE-1, ROUGE-2, ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg_rouge1, agg_rouge2, agg_rougeL = 0, 0, 0

    for h, r in zip(hypotheses, references):
        scores = scorer.score(r, h)  # order: reference, hypothesis
        agg_rouge1 += scores["rouge1"].fmeasure
        agg_rouge2 += scores["rouge2"].fmeasure
        agg_rougeL += scores["rougeL"].fmeasure

    n = len(hypotheses)
    return {
        "rouge1": agg_rouge1 / n,
        "rouge2": agg_rouge2 / n,
        "rougeL": agg_rougeL / n
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to model predictions")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to gold explanations")
    args = parser.parse_args()

    # Read lines
    with open(args.pred_file, "r", encoding="utf-8") as f:
        predictions = [line.strip() for line in f.readlines() if line.strip()]

    with open(args.gold_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f.readlines() if line.strip()]

    # Ensure same length
    assert len(predictions) == len(references), "Mismatch in number of predictions vs. references"

    bleu = compute_bleu(predictions, references)
    rouge_dict = compute_rouge(predictions, references)

    print(f"BLEU: {bleu:.4f}")
    print(f"ROUGE-1: {rouge_dict['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_dict['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_dict['rougeL']:.4f}")

if __name__ == "__main__":
    nltk.download("punkt", quiet=True)  # Ensure NLTK tokenizer is downloaded
    main()
