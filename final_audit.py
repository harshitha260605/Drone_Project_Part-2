from pathlib import Path
import pandas as pd

checks = []

def check(name, condition, detail=""):
    checks.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" +
          (f" — {detail}" if detail else ""))

data_path = Path("data/movement_features_sessions.csv")
metrics_path = Path("results/six_fold_session_metrics.csv")
predictions_path = Path("results/six_fold_session_predictions.csv")
gate_path = Path("results/gate_threshold_sweep.csv")

if data_path.exists():
    df = pd.read_csv(data_path)
    check("Dataset has 306 rows", len(df) == 306, f"rows={len(df)}")
    check("Dataset has 6 sessions",
          df["session_id"].nunique() == 6,
          f"sessions={df['session_id'].nunique()}")
    labels_ok = all(
        set(g["label"].astype(str)) == {"controlled", "varied"}
        for _, g in df.groupby("session_id")
    )
    check("Each session has both labels", labels_ok)
else:
    check("Dataset file exists", False, str(data_path))

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    check("Six-fold metrics saved", len(metrics) == 6,
          f"folds={len(metrics)}")
else:
    check("Six-fold metrics file exists", False, str(metrics_path))

if predictions_path.exists():
    preds = pd.read_csv(predictions_path)
    check("Out-of-fold predictions saved", len(preds) == 306,
          f"predictions={len(preds)}")
else:
    check("Predictions file exists", False, str(predictions_path))

if gate_path.exists():
    gate = pd.read_csv(gate_path)
    check("Gate scenarios saved", len(gate) == 30,
          f"scenarios={len(gate)}")
    if len(gate) == 30:
        baseline = gate["baseline_decision"].value_counts().to_dict()
        decisions = gate["gate_decision"].value_counts().to_dict()
        check("Baseline counts match",
              baseline.get("accept", 0) == 18 and baseline.get("hold", 0) == 12,
              str(baseline))
        check("Gate counts match",
              decisions.get("accept", 0) == 6
              and decisions.get("reduce_sensitivity", 0) == 6
              and decisions.get("hold", 0) == 18,
              str(decisions))
else:
    check("Gate results file exists", False, str(gate_path))

print("\nFINAL AUDIT:", "ALL CHECKS PASSED" if all(x[1] for x in checks)
      else "SOME CHECKS FAILED — review the items above")
