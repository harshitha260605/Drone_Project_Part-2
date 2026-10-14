from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/movement_features_sessions.csv")
FEATURES = [
    "tremor_proxy",
    "trajectory_smoothness",
    "gesture_consistency",
    "velocity_fluctuation",
    "motion_entropy",
]
LABELS = ["controlled", "varied"]

if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}")

df = pd.read_csv(DATA)
required = FEATURES + ["label", "session_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

df = df.dropna(subset=required).reset_index(drop=True)
if df["session_id"].nunique() != 6:
    raise SystemExit(
        f"Expected 6 sessions, found {df['session_id'].nunique()}. "
        "Check the dataset before continuing."
    )

if set(df["label"].astype(str).unique()) != set(LABELS):
    raise SystemExit("Expected both labels: controlled and varied.")

for sid, group in df.groupby("session_id"):
    if set(group["label"].astype(str).unique()) != set(LABELS):
        raise SystemExit(f"Session {sid} does not contain both labels.")

X = df[FEATURES]
y = df["label"].astype(str)
groups = df["session_id"].astype(str)

logo = LeaveOneGroupOut()
all_true = []
all_pred = []
fold_results = []

print("LEAVE-ONE-SESSION-OUT EVALUATION")
print("Each fold tests one complete session.")
print("Dataset rows:", len(df))
print("Sessions:", groups.nunique())
print("Label counts:", y.value_counts().to_dict())

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    held_out = sorted(groups.iloc[test_idx].unique())

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    result = {
        "fold": fold,
        "held_out_session": held_out[0],
        "test_samples": len(test_idx),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_precision": precision_score(
            y_test, pred, labels=LABELS, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, pred, labels=LABELS, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, pred, labels=LABELS, average="macro", zero_division=0
        ),
    }
    fold_results.append(result)
    all_true.extend(y_test.tolist())
    all_pred.extend(pred.tolist())

    print(
        f"\nFold {fold}/6 | Held-out session: {held_out[0]} "
        f"| n={len(test_idx)}"
    )
    print(f"  Accuracy:          {result['accuracy']:.3f}")
    print(f"  Balanced accuracy: {result['balanced_accuracy']:.3f}")
    print(f"  Macro F1:          {result['macro_f1']:.3f}")

results = pd.DataFrame(fold_results)
print("\nPER-FOLD RESULTS")
print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\nMEAN AND STANDARD DEVIATION ACROSS SIX FOLDS")
for metric in [
    "accuracy", "balanced_accuracy", "macro_precision",
    "macro_recall", "macro_f1"
]:
    print(
        f"{metric}: {results[metric].mean():.3f} "
        f"+/- {results[metric].std(ddof=1):.3f}"
    )

print("\nAGGREGATED OUT-OF-FOLD RESULTS")
print(classification_report(
    all_true, all_pred, labels=LABELS, zero_division=0
))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(all_true, all_pred, labels=LABELS))

out = Path("results")
out.mkdir(exist_ok=True)
results.to_csv(out / "six_fold_session_metrics.csv", index=False)
pd.DataFrame({
    "true_label": all_true,
    "predicted_label": all_pred,
}).to_csv(out / "six_fold_session_predictions.csv", index=False)

print("\nSaved fold metrics to results/six_fold_session_metrics.csv")
print("Saved predictions to results/six_fold_session_predictions.csv")
print("\nLIMITATIONS:")
print("- Labels indicate controlled/varied movement protocols only.")
print("- Sessions from the same person do not establish new-person generalization.")
print("- Overlapping windows within sessions may be correlated.")
print("- This does not evaluate cognitive state or real-drone safety.")
