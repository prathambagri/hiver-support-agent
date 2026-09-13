"""
Run this AFTER you've finished labelling data/golden_set.xlsx.
Converts it back to data/golden_set.csv (overwriting it) so evaluate.py can read it.

Run: python xlsx_to_csv_after_labelling.py
"""
import pandas as pd
import config as C

SRC = C.GOLDEN_CSV.replace(".csv", ".xlsx")
OUT = C.GOLDEN_CSV


def main():
    df = pd.read_excel(SRC)
    df.to_csv(OUT, index=False, encoding="utf-8")
    labelled = df["gold_intent"].notna().sum()
    print(f"Saved -> {OUT}  ({labelled}/{len(df)} rows have gold_intent filled in)")
    if labelled < 150:
        print("Fewer than 150 labelled - spec wants 150-250. Keep going before running evaluate.py.")


if __name__ == "__main__":
    main()
