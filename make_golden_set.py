"""
Step 4a: Produce the CSV you will hand-label to create the golden evaluation set
(150-250 examples per spec). This samples from the SAME test_sample the agent/baselines
ran on, so predictions line up 1:1 with gold labels later.

Run: python make_golden_set.py
Then open data/golden_set.csv and fill in gold_intent + gold_escalate + notes by hand.
"""
import pandas as pd
import config as C

VALID_INTENTS_COMMENT = f"# valid intents: {', '.join(C.INTENTS)}"


def main():
    test = pd.read_parquet(C.TEST_PARQUET)
    n = min(C.GOLDEN_SET_SIZE, len(test))
    sample = test.sample(n=n, random_state=C.RANDOM_SEED).reset_index(drop=True)

    out = pd.DataFrame({
        "customer_tweet_id": sample["customer_tweet_id"],
        "customer_text": sample["customer_text"],
        "actual_brand_reply_from_dataset": sample["brand_reply_text"],  # useful reference while labelling, NOT ground truth for reply quality
        "gold_intent": "",          # <-- fill by hand, one of C.INTENTS
        "gold_escalate": "",        # <-- fill by hand: TRUE / FALSE
        "gold_escalate_notes": "",  # <-- optional: why you'd escalate or not
    })

    out.to_csv(C.GOLDEN_CSV, index=False)
    print(f"Wrote {len(out)} rows to {C.GOLDEN_CSV}")
    print(VALID_INTENTS_COMMENT)
    print("\nHOW TO LABEL FAST (aim for <20 sec/row):")
    print("1. Read customer_text only (ignore actual_brand_reply_from_dataset for intent - "
          "it's just there so you don't have to re-derive context; don't peek before deciding intent).")
    print("2. Pick the ONE intent from the fixed list that fits best.")
    print("3. Decide gold_escalate using the same rule you'd want an agent to use "
          "(see ESCALATION_GUIDANCE in config.py) - be consistent, not perfect.")
    print("4. Leave gold_escalate_notes blank unless something's ambiguous - keep moving.")


if __name__ == "__main__":
    main()
