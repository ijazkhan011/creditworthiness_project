"""
Load the best saved model and make predictions for new applicants.
Usage:
    python src/infer.py
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best", help="best or a specific model file name")
    parser.add_argument("--records", type=str, help="JSON string with a list of applicants")
    args = parser.parse_args()

    if args.model == "best":
        # find a file that starts with best_model_
        best_files = list(RESULTS_DIR.glob("best_model_*.joblib"))
        if not best_files:
            raise FileNotFoundError("No best model found. Train models first.")
        model_path = best_files[0]
    else:
        model_path = RESULTS_DIR / args.model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    pipe = joblib.load(model_path)

    # Example default record if none provided
    default_records = [
        {
            "income": 2_000_000,
            "age": 28,
            "debt_to_income": 0.35,
            "num_open_credit_lines": 4,
            "delinquencies_last_2y": 0,
            "credit_history_years": 3.5,
            "avg_payment_delay_days": 1.0,
            "has_mortgage": 0,
            "has_car_loan": 1,
            "savings_balance": 200_000,
        }
    ]

    records = json.loads(args.records) if args.records else default_records
    X_new = pd.DataFrame(records)
    proba = pipe.predict_proba(X_new)[:, 1]
    pred = (proba >= 0.5).astype(int)

    for i, (p, pr) in enumerate(zip(pred, proba)):
        print(f"Record {i}: creditworthy={int(p)} (prob={pr:.3f})")

if __name__ == "__main__":
    main()
