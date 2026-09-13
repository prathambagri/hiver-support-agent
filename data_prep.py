"""
Step 1: Load raw twcs.csv, reconstruct (customer_message -> brand_reply) pairs for one
brand, then split into:
  - corpus: historical resolved pairs used for retrieval grounding
  - test_sample: held-out customer messages the agent will actually process

Raw schema (Kaggle thoughtvector/customer-support-on-twitter, file twcs.csv):
  tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

Run: python data_prep.py
"""
import pandas as pd
import numpy as np
import config as C


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tweet_id": str, "author_id": str,
                                   "response_tweet_id": str, "in_response_to_tweet_id": str})
    return df


def reconstruct_pairs(df: pd.DataFrame, brand: str = C.BRAND_HANDLE) -> pd.DataFrame:
    """Build (customer_text, brand_text) pairs: a brand reply joined to the inbound
    tweet it was replying to."""
    df = df.copy()
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])

    brand_replies = df[(~df["inbound"]) & (df["author_id"] == brand)]
    customer_tweets = df[df["inbound"]]

    merged = brand_replies.merge(
        customer_tweets,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_brand", "_customer"),
    )

    pairs = merged[[
        "tweet_id_customer", "text_customer", "created_at_customer",
        "tweet_id_brand", "text_brand", "created_at_brand",
    ]].rename(columns={
        "tweet_id_customer": "customer_tweet_id",
        "text_customer": "customer_text",
        "created_at_customer": "customer_created_at",
        "tweet_id_brand": "brand_tweet_id",
        "text_brand": "brand_reply_text",
        "created_at_brand": "brand_created_at",
    })

    pairs = pairs.drop_duplicates(subset=["customer_tweet_id"]).reset_index(drop=True)
    return pairs


def split_corpus_and_test(pairs: pd.DataFrame, seed=C.RANDOM_SEED):
    pairs = pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = min(C.TEST_SAMPLE_SIZE, len(pairs) // 4)  # never let test eat >25% of data
    test = pairs.iloc[:n_test].copy()
    corpus_pool = pairs.iloc[n_test:].copy()
    corpus = corpus_pool.iloc[: C.CORPUS_MAX_SIZE].copy()
    return corpus, test


def main():
    print(f"Loading raw data from {C.RAW_CSV} ...")
    df = load_raw()
    print(f"Raw rows: {len(df):,}")

    pairs = reconstruct_pairs(df, C.BRAND_HANDLE)
    print(f"Reconstructed {len(pairs):,} customer<->brand pairs for {C.BRAND_HANDLE}")

    if len(pairs) < 500:
        print("WARNING: very few pairs found. Double check BRAND_HANDLE in config.py "
              "matches an author_id in the dataset exactly (case-sensitive).")

    corpus, test = split_corpus_and_test(pairs)
    print(f"Corpus (grounding pool): {len(corpus)} | Test sample (agent runs on this): {len(test)}")

    corpus.to_parquet(C.CORPUS_PARQUET, index=False)
    test.to_parquet(C.TEST_PARQUET, index=False)
    print(f"Saved -> {C.CORPUS_PARQUET}, {C.TEST_PARQUET}")


if __name__ == "__main__":
    main()
