import pandas as pd
import os

def process_file(input_path, output_dir, split_name):
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["Sentence1", "Sentence2", "gold_label", "Explanation_1"])
    df = df[df["gold_label"].isin(["entailment", "neutral", "contradiction"])]
    df["Explanation"] = df["Explanation_1"].str.strip()
    df = df[df["Explanation_1"].str.len() > 10]

    explainer_df = df[["Sentence1", "Sentence2", "Explanation_1"]].rename(
        columns={"Sentence1": "premise", "Sentence2": "hypothesis"}
    )
    explainer_df.to_csv(os.path.join(output_dir, f"explainer_{split_name}.csv"), index=False)

    predictor_df = df[["Sentence1", "Sentence2", "Explanation_1", "gold_label"]].rename(
        columns={"Sentence1": "premise", "Sentence2": "hypothesis", "gold_label": "label"}
    )
    predictor_df.to_csv(os.path.join(output_dir, f"predictor_{split_name}.csv"), index=False)

    print(f"[{split_name}] Saved {len(explainer_df)} examples")

def preprocess_all():
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Training data is in two parts
    process_file("data/esnli_train_1.csv", output_dir, "train_1")
    process_file("data/esnli_train_2.csv", output_dir, "train_2")
    process_file("data/esnli_dev.csv", output_dir, "dev")
    process_file("data/esnli_test.csv", output_dir, "test")

if __name__ == "__main__":
    preprocess_all()
