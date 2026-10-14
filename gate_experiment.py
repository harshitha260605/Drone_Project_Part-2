import csv
from collections import Counter
from pathlib import Path

from drone_safety.decision_fusion import (
    DecisionResult,
    DecisionConfig,
    validate_command,
)

CONFIDENCES = [0.50, 0.74, 0.75, 0.90, 1.00]
STABILITIES = [0.30, 0.59, 0.60, 0.71, 0.72, 0.90]

config = DecisionConfig()
rows = []
baseline_counts = Counter()
gate_counts = Counter()

for confidence in CONFIDENCES:
    for stability in STABILITIES:
        baseline = (
            "accept"
            if confidence >= config.min_gesture_confidence
            else "hold"
        )

        decision = validate_command(
            "MOVE_FORWARD",
            confidence,
            stability,
            config,
        )

        gate = decision.result.value
        baseline_counts[baseline] += 1
        gate_counts[gate] += 1

        rows.append({
            "gesture_confidence": confidence,
            "stability_score": stability,
            "baseline_decision": baseline,
            "gate_decision": gate,
            "gate_reason": decision.reason,
            "simulated_command": decision.simulated_command or "",
            "baseline_accepts": baseline == "accept",
            "gate_fully_accepts": gate == "accept",
            "gate_reduces_sensitivity": gate == "reduce_sensitivity",
            "gate_holds": gate == "hold",
        })

# Verify important threshold boundaries.
assert validate_command("MOVE_FORWARD", 0.75, 0.72).result == DecisionResult.ACCEPT
assert validate_command("MOVE_FORWARD", 0.75, 0.71).result == DecisionResult.REDUCE_SENSITIVITY
assert validate_command("MOVE_FORWARD", 0.75, 0.60).result == DecisionResult.REDUCE_SENSITIVITY
assert validate_command("MOVE_FORWARD", 0.75, 0.59).result == DecisionResult.HOLD
assert validate_command("MOVE_FORWARD", 0.74, 0.90).result == DecisionResult.HOLD

output = Path("results")
output.mkdir(exist_ok=True)
csv_path = output / "gate_threshold_sweep.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

total = len(rows)
print("SYNTHETIC SAFETY-GATE COMPARISON")
print("Scenario type: predefined confidence/stability combinations")
print(f"Total scenarios: {total}")
print(f"Boundary assertions: PASS")
print(f"\nConfidence-only baseline:")
print(f"  Accept: {baseline_counts['accept']} ({baseline_counts['accept']/total:.1%})")
print(f"  Hold:   {baseline_counts['hold']} ({baseline_counts['hold']/total:.1%})")

print("\nStability-aware gate:")
for result in ["accept", "reduce_sensitivity", "hold"]:
    count = gate_counts[result]
    print(f"  {result}: {count} ({count/total:.1%})")

baseline_accepts = sum(r["baseline_accepts"] for r in rows)
fully_accepted = sum(r["gate_fully_accepts"] for r in rows)
reduced = sum(r["gate_reduces_sensitivity"] for r in rows)
held = sum(r["gate_holds"] for r in rows)

print("\nAmong baseline-accepted scenarios:")
print(f"  Fully accepted by gate: {sum(r['baseline_decision'] == 'accept' and r['gate_decision'] == 'accept' for r in rows)}")
print(f"  Reduced sensitivity:    {sum(r['baseline_decision'] == 'accept' and r['gate_decision'] == 'reduce_sensitivity' for r in rows)}")
print(f"  Held by gate:           {sum(r['baseline_decision'] == 'accept' and r['gate_decision'] == 'hold' for r in rows)}")

print(f"\nScenario-level results saved to: {csv_path}")
print("\nLIMITATION: Synthetic threshold behavior only.")
print("This does not measure real-world safety or drone-flight performance.")
