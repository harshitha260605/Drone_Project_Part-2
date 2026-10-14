import pandas as pd

df = pd.read_csv("data/movement_features_sessions.csv")

print("LABEL COUNTS PER SESSION")
print(pd.crosstab(df["session_id"], df["label"]).to_string())

print("\nFEATURE SUMMARY BY LABEL")
features = [
    "tremor_proxy",
    "trajectory_smoothness",
    "gesture_consistency",
    "velocity_fluctuation",
    "motion_entropy",
]
print(df.groupby("label")[features].mean().round(4).to_string())
