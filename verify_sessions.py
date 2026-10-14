import csv
from collections import Counter, defaultdict
from pathlib import Path

path = Path("data/movement_features_sessions.csv")

if not path.exists():
    print("ERROR: Dataset file not found:", path)
    raise SystemExit(1)

with path.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("ERROR: Dataset is empty.")
    raise SystemExit(1)

sessions = defaultdict(list)
for row in rows:
    sessions[row.get("session_id", "").strip()].append(row)

print("\n========== DATASET VERIFICATION ==========")
print("File:", path)
print("Total rows:", len(rows))
print("Unique session IDs:", len(sessions))
print("Overall label counts:", dict(Counter(r["label"] for r in rows)))

print("\n========== EACH SESSION ==========")
for sid, session_rows in sorted(sessions.items()):
    labels = Counter(r["label"] for r in session_rows)
    print(f"\nSession ID: {sid}")
    print("  Samples:", len(session_rows))
    print("  Labels:", dict(labels))
    print("  Has both labels:", "controlled" in labels and "varied" in labels)

print("\n========== DATA QUALITY ==========")
print("Blank session IDs:", sum(not r.get("session_id", "").strip() for r in rows))
print("Unexpected labels:", sorted({
    r.get("label", "") for r in rows
    if r.get("label", "") not in {"controlled", "varied"}
}))
print("Rows with missing feature values:", sum(
    any(not r.get(col, "").strip() for col in [
        "tremor_proxy", "trajectory_smoothness",
        "gesture_consistency", "velocity_fluctuation",
        "motion_entropy", "frame_count"
    ])
    for r in rows
))

print("\n========== VERDICT ==========")
print("Six sessions present:", len(sessions) == 6 and "" not in sessions)
print("Check the counts above before model evaluation.")
