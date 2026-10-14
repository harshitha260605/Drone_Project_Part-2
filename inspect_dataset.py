import pandas as pd

for path in [
    "data/movement_features.csv",
    "data/movement_features_sessions.csv",
]:
    print("\n" + "=" * 55)
    print("FILE:", path)
    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Missing values:", int(df.isna().sum().sum()))

    if "label" in df.columns:
        print("\nLabel counts:")
        print(df["label"].value_counts(dropna=False).to_string())

    if "session_id" in df.columns:
        print("\nUnique sessions:", df["session_id"].nunique())
        print("Rows per session:")
        print(df.groupby("session_id").size().to_string())
