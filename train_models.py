"""
Train creditworthiness models and save the best model.
Usage:
    python src/train_models.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "credit_data.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["creditworthy"])
    y = df["creditworthy"]
    numeric_features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    preprocess = ColumnTransformer([("num", StandardScaler(), numeric_features)], remainder="drop")

    models = {
        "logreg": LogisticRegression(max_iter=200),
        "dtree": DecisionTreeClassifier(random_state=42),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    }

    metrics = []
    roc_curves = {}

    for name, clf in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        aucv = roc_auc_score(y_test, y_proba)

        metrics.append({"model": name, "precision": prec, "recall": rec, "f1": f1, "roc_auc": aucv})

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, aucv)

        joblib.dump(pipe, RESULTS_DIR / f"model_{name}.joblib")

    metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    # ROC plot
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, aucv) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name.upper()} (AUC={aucv:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Creditworthiness Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # Save best model
    best_model = metrics_df.iloc[0]["model"]
    best_path = RESULTS_DIR / f"best_model_{best_model}.joblib"
    joblib.dump(joblib.load(RESULTS_DIR / f"model_{best_model}.joblib"), best_path)

    print("Training complete. Best model:", best_model)
    print(metrics_df)

if __name__ == "__main__":
    main()
