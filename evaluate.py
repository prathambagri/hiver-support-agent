"""
Step 4b: The evaluation harness. Run AFTER you've hand-filled data/golden_set.csv.

Computes, for agent + both baselines, on the golden set only:
  - intent classification accuracy + macro F1
  - escalation precision/recall/F1 (escalate=positive class - the imbalanced, informative one)
  - LLM-judge reply quality score (1-5)
  - judge-vs-human agreement (Spearman correlation) on a small human-scored subset

Run: python evaluate.py
"""
import json
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import spearmanr

import config as C
from llm_judge import judge_reply


def load_jsonl(path) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path)]
    return pd.DataFrame(rows)


def to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


def score_predictions(name: str, preds: pd.DataFrame, golden: pd.DataFrame) -> dict:
    golden = golden.copy()
    preds = preds.copy()
    golden["customer_tweet_id"] = golden["customer_tweet_id"].astype(str)
    preds["customer_tweet_id"] = preds["customer_tweet_id"].astype(str)
    merged = golden.merge(preds, on="customer_tweet_id", suffixes=("_gold", "_pred"))
    if len(merged) == 0:
        raise ValueError(f"No overlap between golden set and {name} predictions - "
                          f"did you run agent.py/baselines.py on the same test_sample?")

    gold_intent = merged["gold_intent"]
    pred_intent = merged["intent"]
    intent_acc = accuracy_score(gold_intent, pred_intent)
    intent_f1 = f1_score(gold_intent, pred_intent, average="macro", zero_division=0)

    gold_esc = merged["gold_escalate"].map(to_bool)
    pred_esc = merged["escalate"].map(to_bool)
    esc_precision = precision_score(gold_esc, pred_esc, zero_division=0)
    esc_recall = recall_score(gold_esc, pred_esc, zero_division=0)
    esc_f1 = f1_score(gold_esc, pred_esc, zero_division=0)
    esc_accuracy = accuracy_score(gold_esc, pred_esc)  # kept for the report's "misleading headline" section

    return {
        "name": name,
        "n": len(merged),
        "intent_accuracy": intent_acc,
        "intent_macro_f1": intent_f1,
        "escalate_accuracy": esc_accuracy,
        "escalate_precision": esc_precision,
        "escalate_recall": esc_recall,
        "escalate_f1": esc_f1,
        "_merged": merged,  # kept internally for the judge step below
    }


def run_judge_on_replies(merged: pd.DataFrame, n_sample: int = None) -> pd.DataFrame:
    rows = merged if n_sample is None else merged.sample(n=min(n_sample, len(merged)), random_state=C.RANDOM_SEED)
    scores = []
    for _, row in rows.iterrows():
        result = judge_reply(row["customer_text_gold"], row["reply"])
        scores.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text_gold"],
            "reply": row["reply"],
            "judge_score": result["score"],
            "judge_justification": result["justification"],
        })
        time.sleep(4.5)  # stay under gemini-3.5-flash-lite's 15 req/min free-tier limit
    return pd.DataFrame(scores)


def judge_human_agreement(judge_df: pd.DataFrame, human_scores: dict) -> float:
    """human_scores: {customer_tweet_id: your_1_to_5_score}. Returns Spearman correlation."""
    df = judge_df[judge_df["customer_tweet_id"].isin(human_scores)].copy()
    df["human_score"] = df["customer_tweet_id"].map(human_scores)
    if len(df) < 3:
        return float("nan")
    corr, _ = spearmanr(df["judge_score"], df["human_score"])
    return corr


def main():
    golden = pd.read_csv(C.GOLDEN_CSV)
    golden = golden[golden["gold_intent"].notna() & (golden["gold_intent"] != "")]
    if len(golden) < 50:
        print(f"WARNING: only {len(golden)} labelled rows found in {C.GOLDEN_CSV}. "
              f"Fill it in before trusting these numbers (spec wants 150-250).")

    runs = {
        "agent": C.AGENT_OUT,
        "baseline_trivial": C.BASELINE_TRIVIAL_OUT,
        "baseline_simple": C.BASELINE_SIMPLE_OUT,
    }

    summary_rows = []
    for name, path in runs.items():
        preds = load_jsonl(path)
        result = score_predictions(name, preds, golden)
        merged = result.pop("_merged")

        # LLM-judge on a manageable subset of replies (all 3 systems, same rows, for fair comparison)
        judge_df = run_judge_on_replies(merged, n_sample=min(40, len(merged)))
        result["mean_judge_score"] = judge_df["judge_score"].dropna().mean()
        judge_df.to_csv(f"outputs/judge_scores_{name}.csv", index=False)

        summary_rows.append(result)
        print(f"\n=== {name} (n={result['n']}) ===")
        print(f"  intent accuracy:      {result['intent_accuracy']:.3f}")
        print(f"  intent macro F1:      {result['intent_macro_f1']:.3f}")
        print(f"  escalate accuracy:    {result['escalate_accuracy']:.3f}  (often misleading - see report)")
        print(f"  escalate precision:   {result['escalate_precision']:.3f}")
        print(f"  escalate recall:      {result['escalate_recall']:.3f}")
        print(f"  escalate F1:          {result['escalate_f1']:.3f}")
        print(f"  mean LLM-judge score: {result['mean_judge_score']:.2f} / 5")

    pd.DataFrame(summary_rows).drop(columns=["_merged"], errors="ignore").to_csv(
        "outputs/summary_metrics.csv", index=False)
    print("\nSaved -> outputs/summary_metrics.csv, outputs/judge_scores_<name>.csv")
    print("\nNEXT: open outputs/judge_scores_agent.csv, hand-score ~30 of those replies yourself "
          "1-5 in a new column, then call judge_human_agreement() (see bottom of this file) "
          "to get the correlation number the report needs.")


if __name__ == "__main__":
    main()
