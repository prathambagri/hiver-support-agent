"""
Step 3: Baselines to compare the agent against (spec requires >=2: trivial + simple).

Run: python baselines.py
"""
import json
import os
import re
import pandas as pd
import config as C
from retrieval import HistoricalRetriever

# ---------- Baseline 1: trivial ----------
# Always predicts the majority intent, always the same generic reply, never escalates.
TRIVIAL_REPLY = ("Hi, thanks for reaching out! We've received your message and a member "
                  "of our team will follow up with you shortly.")


def trivial_baseline(test: pd.DataFrame, majority_intent: str) -> list:
    results = []
    for _, row in test.iterrows():
        results.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text"],
            "intent": majority_intent,
            "reply": TRIVIAL_REPLY,
            "escalate": False,
            "escalate_reason": "trivial baseline never escalates",
        })
    return results


# ---------- Baseline 2: simple rule-based ----------
KEYWORD_RULES = {
    "delivery_or_shipping_issue": ["deliver", "shipping", "shipment", "tracking", "package", "arrive", "late"],
    "refund_or_return": ["refund", "return", "money back", "reimburse"],
    "order_cancellation": ["cancel"],
    "account_or_login_issue": ["login", "log in", "password", "account locked", "can't access", "sign in"],
    "billing_or_payment_issue": ["charge", "billed", "payment", "double charged", "invoice"],
    "product_defect_or_wrong_item": ["broken", "defect", "wrong item", "damaged", "doesn't work", "faulty"],
    "compliment_or_other": ["thank you", "thanks", "great job", "love"],
}

ESCALATE_KEYWORDS = ["lawyer", "legal action", "fraud", "chargeback", "dispute",
                      "hacked", "unauthorized", "still not fixed", "third time", "again and again"]


def rule_based_intent(text: str) -> str:
    t = text.lower()
    for intent, kws in KEYWORD_RULES.items():
        if any(kw in t for kw in kws):
            return intent
    return "general_inquiry"


def rule_based_escalate(text: str) -> tuple:
    t = text.lower()
    for kw in ESCALATE_KEYWORDS:
        if kw in t:
            return True, f"matched escalation keyword: '{kw}'"
    return False, "no escalation keyword matched"


def simple_baseline(test: pd.DataFrame, retriever: HistoricalRetriever) -> list:
    results = []
    for _, row in test.iterrows():
        text = row["customer_text"]
        intent = rule_based_intent(text)
        escalate, reason = rule_based_escalate(text)

        # Template reply = the closest historical resolved reply for that intent-ish query.
        similar = retriever.top_k(text, k=1)
        reply = similar[0]["past_brand_reply"] if similar else TRIVIAL_REPLY

        results.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": text,
            "intent": intent,
            "reply": reply,
            "escalate": escalate,
            "escalate_reason": reason,
        })
    return results


def write_jsonl(records: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    test = pd.read_parquet(C.TEST_PARQUET)
    corpus = pd.read_parquet(C.CORPUS_PARQUET)
    retriever = HistoricalRetriever(corpus)

    # crude majority intent guess for the trivial baseline: whichever rule fires most on the corpus
    corpus_intents = corpus["customer_text"].fillna("").map(rule_based_intent)
    majority_intent = corpus_intents.value_counts().idxmax()
    print(f"Majority intent (trivial baseline): {majority_intent}")

    trivial = trivial_baseline(test, majority_intent)
    write_jsonl(trivial, C.BASELINE_TRIVIAL_OUT)

    simple = simple_baseline(test, retriever)
    write_jsonl(simple, C.BASELINE_SIMPLE_OUT)

    print(f"Saved -> {C.BASELINE_TRIVIAL_OUT}, {C.BASELINE_SIMPLE_OUT}")


if __name__ == "__main__":
    main()
