#this is to be changed
#this is used to compare performance of model in terms of predicted label given premise+hypothesis vs premise+hypoethesis+explanation

import subprocess

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    # 1) Train "with explanation" model
    run_command("python scripts/train_predictor.py --train_csv data/processed/predictor_train.csv --dev_csv data/processed/predictor_dev.csv --output_dir checkpoints/predictor_with_expl")

    # 2) Train "without explanation" model (using ablation dataset or code changes)
    run_command("python scripts/train_predictor.py --train_csv data/processed/predictor_train_no_expl.csv --dev_csv data/processed/predictor_dev_no_expl.csv --output_dir checkpoints/predictor_no_expl")

    # 3) Evaluate both
    # You can re-run 'evaluate' or do it inside the script. Then parse results, etc.
    # In a real scenario, you'd do something like parse logs, gather final dev accuracy, then compare.
    print("Ablation done! Compare logs for final accuracy.")
