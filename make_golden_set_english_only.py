"""
Replaces make_golden_set.py's output with an ENGLISH-ONLY version, sampled from the
same test_sample.parquet the agent/baselines already ran on (so predictions still
line up). This is a deliberate scope decision: manual labelling by a single
non-multilingual annotator is only reliable in one language - document this in
DECISION_LOG.md and the report's "what I chose not to build" section.

Run: python make_golden_set_english_only.py
"""
import pandas as pd
from langdetect import detect, DetectorFactory
import config as C

DetectorFactory.seed = 42  # deterministic langdetect output


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def main():
    test = pd.read_parquet(C.TEST_PARQUET)
    print(f"Test sample size: {len(test)}")

    test["is_en"] = test["customer_text"].fillna("").map(is_english)
    english_rows = test[test["is_en"]].drop(columns=["is_en"])
    print(f"English-language rows available: {len(english_rows)}")

    n = min(C.GOLDEN_SET_SIZE, len(english_rows))
    if n < 150:
        print(f"WARNING: only {n} English rows available, spec wants 150-250. "
              f"Lower GOLDEN_SET_SIZE expectations or tell Claude - we can pull a bigger "
              f"English-only test sample by re-running data_prep.py + agent.py with a "
              f"language filter baked in earlier in the pipeline.")

    sample = english_rows.sample(n=n, random_state=C.RANDOM_SEED).reset_index(drop=True)

    out = pd.DataFrame({
        "customer_tweet_id": sample["customer_tweet_id"],
        "customer_text": sample["customer_text"],
        "actual_brand_reply_from_dataset": sample["brand_reply_text"],
        "gold_intent": "",
        "gold_escalate": "",
        "gold_escalate_notes": "",
    })

    out.to_csv(C.GOLDEN_CSV, index=False)
    print(f"Wrote {len(out)} ENGLISH-ONLY rows to {C.GOLDEN_CSV}")
    print("Now run: python csv_to_xlsx_for_labelling.py  (to regenerate the readable xlsx)")


if __name__ == "__main__":
    main()
