"""
Step 4c: After running evaluate.py, open outputs/judge_scores_agent.csv, add a
'human_score' column and fill in ~30 rows yourself (1-5, same rubric as llm_judge.py).
Then run this script to get the correlation number the report needs.

Run: python agreement_check.py
"""
import pandas as pd
from scipy.stats import spearmanr
import config as C

PATH = "outputs/judge_scores_agent.csv"  # add your human_score column into this file


def main():
    df = pd.read_csv(PATH)
    df = df[df["human_score"].notna()]
    if len(df) < 10:
        print(f"Only {len(df)} human-scored rows found - aim for at least "
              f"{C.JUDGE_AGREEMENT_SAMPLE} for a defensible agreement number.")
    corr, p = spearmanr(df["judge_score"], df["human_score"])
    print(f"n={len(df)}  Spearman correlation (judge vs human) = {corr:.3f}  (p={p:.3f})")
    print("Report this number + n in the report's evaluation-harness section.")


if __name__ == "__main__":
    main()
