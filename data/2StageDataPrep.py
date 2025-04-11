import pandas as pd
import os

def process_dataframe(df, output_dir, split_name):
    # Drop rows with missing important fields
    df = df.dropna(subset=["Sentence1", "Sentence2", "gold_label", "Explanation_1"])
    
    # Keep only valid labels
    df = df[df["gold_label"].isin(["entailment", "neutral", "contradiction"])]
    
    # Clean explanations
    df["Explanation"] = df["Explanation_1"].str.strip()
    df = df[df["Explanation_1"].str.len() > 10]

    # Explainer format
    explainer_df = df[["Sentence1", "Sentence2", "Explanation_1"]].rename(
        columns={"Sentence1": "premise", "Sentence2": "hypothesis", "Explanation_1": "explanation"}
    )
    explainer_df.to_csv(os.path.join(output_dir, f"explainer_{split_name}.csv"), index=False)

    # Predictor format
    predictor_df = df[["Sentence1", "Sentence2", "Explanation_1", "gold_label"]].rename(
        columns={"Sentence1": "premise", "Sentence2": "hypothesis", "Explanation_1": "explanation", "gold_label": "label"}
    )
    predictor_df.to_csv(os.path.join(output_dir, f"predictor_{split_name}.csv"), index=False)

    print(f"[{split_name}] Saved {len(explainer_df)} examples")

def process_file(input_path, output_dir, split_name):
    df = pd.read_csv(input_path)
    process_dataframe(df, output_dir, split_name)

def preprocess_all():
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Combine both training splits
    train_df1 = pd.read_csv("data/esnli_train_1.csv")
    train_df2 = pd.read_csv("data/esnli_train_2.csv")
    combined_train = pd.concat([train_df1, train_df2], ignore_index=True)
    process_dataframe(combined_train, output_dir, "train")

    # ✅ Process dev and test individually
    process_file("data/esnli_dev.csv", output_dir, "dev")
    process_file("data/esnli_test.csv", output_dir, "test")

if __name__ == "__main__":
    preprocess_all()
