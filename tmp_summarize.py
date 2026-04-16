import pandas as pd
import glob

print("--- UM Classification Results ---")
for f in glob.glob("results/um/*_summary.csv"):
    if "classification" in f:
        try:
            df = pd.read_csv(f)
            print(f)
            print(df.to_string())
        except Exception as e:
            pass

print("\n--- UM Regression Results ---")
for f in glob.glob("results/um/*_summary.csv"):
    if "regression" in f:
        try:
            df = pd.read_csv(f)
            print(f)
            print(df.to_string())
        except Exception as e:
            pass
