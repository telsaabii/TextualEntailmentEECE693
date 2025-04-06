#this file will be used to visualize the attention (to be changed)
import os
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from transformers import T5ForConditionalGeneration, T5Tokenizer

def inspect_attention(
    model,
    tokenizer,
    premise,
    hypothesis,
    layer=-1,
    device="cpu"
):
    """
    Generate a dummy explanation forward pass to extract cross-attentions,
    then returns a matplotlib figure object (heatmap).
    """
    input_text = f"explain premise: {premise} hypothesis: {hypothesis}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = tokenizer("explanation", return_tensors="pt").input_ids.to(device)  # dummy target

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels,
        output_attentions=True,
        return_dict=True
    )

    cross_attn = outputs.cross_attentions[layer]  # shape: (batch, heads, tgt_len, src_len)
    avg_attn = cross_attn.mean(dim=1).squeeze(0).detach().cpu().numpy()

    src_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tgt_tokens = tokenizer.convert_ids_to_tokens(labels[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(avg_attn, cmap='viridis')

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(tgt_tokens)

    ax.set_xlabel("Source (Premise+Hypothesis) Tokens")
    ax.set_ylabel("Target (Explanation) Tokens")
    ax.set_title(f"Cross-Attention (Layer {layer})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/explainer",
                        help="Path to your fine-tuned T5 explainer")
    parser.add_argument("--samples_file", type=str, required=True,
                        help="File with lines of 'premise ||| hypothesis'")
    parser.add_argument("--output_pdf", type=str, default="attention_visuals.pdf",
                        help="Name of the output PDF file")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which cross-attention layer to visualize")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)

    pdf = PdfPages(args.output_pdf)

    with open(args.samples_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(lines):
        # Expect 'premise ||| hypothesis'
        premise, hypothesis = line.split("|||")
        fig = inspect_attention(model, tokenizer, premise, hypothesis, layer=args.layer, device=device)
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"Saved attention heatmaps to {args.output_pdf}")

if __name__ == "__main__":
    main()
