"""
Step 2: The actual AI support agent.

For each customer message: retrieve similar past resolutions, then one Gemini call
returns {intent, reply, escalate, escalate_reason} as JSON.

Requires: pip install google-generativeai
Set env var GEMINI_API_KEY before running.

Run: python agent.py
"""
import os
import json
import time
import pandas as pd
import google.generativeai as genai

import config as C
from retrieval import HistoricalRetriever

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(C.GEMINI_MODEL)

SYSTEM_PROMPT = f"""You are a customer support triage agent for {C.BRAND_HANDLE}.

Given a customer's message and up to 3 similar past cases (with how the brand actually
replied), you must return STRICT JSON with exactly these keys:
  "intent": one of {C.INTENTS}
  "reply": a draft reply to the customer, grounded in the tone/content of the past
           replies shown to you where relevant. Do not invent policy details not
           supported by the examples or common sense.
  "escalate": true or false
  "escalate_reason": a short (<20 words) reason for the escalate decision

Escalation policy:
{C.ESCALATION_GUIDANCE}

Return ONLY the JSON object, no markdown fences, no commentary.
"""


def build_prompt(customer_text: str, similar_cases: list) -> str:
    examples = "\n".join(
        f"- Past customer message: {c['past_customer_text']!r}\n"
        f"  Past brand reply: {c['past_brand_reply']!r}"
        for c in similar_cases
    ) or "(no similar past cases found)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"SIMILAR PAST CASES:\n{examples}\n\n"
        f"CUSTOMER MESSAGE:\n{customer_text!r}\n\n"
        f"JSON:"
    )


def call_agent(customer_text: str, retriever: HistoricalRetriever, retries: int = 4) -> dict:
    similar = retriever.top_k(customer_text, k=3)
    prompt = build_prompt(customer_text, similar)

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(prompt)
            raw = resp.text.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            parsed["intent"] = parsed.get("intent") if parsed.get("intent") in C.INTENTS else "general_inquiry"
            parsed["escalate"] = bool(parsed.get("escalate", False))
            parsed.setdefault("escalate_reason", "")
            parsed.setdefault("reply", "")
            return parsed
        except Exception as e:  # noqa: BLE001 - want to retry on any parse/API error
            last_err = e
            wait = 20 if "ResourceExhausted" in type(e).__name__ or "429" in str(e) else 1.5
            print(f"  [retry {attempt+1}/{retries}] agent call failed: {type(e).__name__} - waiting {wait}s")
            time.sleep(wait)
    # Fail-safe fallback so one bad row doesn't kill the whole run
    print(f"  [FALLBACK] giving up on this row after retries: {last_err}")
    return {"intent": "general_inquiry", "reply": "", "escalate": True,
            "escalate_reason": f"agent_error: {last_err}"}


def main():
    test = pd.read_parquet(C.TEST_PARQUET)
    retriever = HistoricalRetriever.from_parquet(C.CORPUS_PARQUET)

    # Only run on rows that are actually in the golden set - no point spending scarce
    # API quota on the ~100 test rows that never get evaluated.
    if os.path.exists(C.GOLDEN_CSV):
        golden_ids = set(pd.read_csv(C.GOLDEN_CSV)["customer_tweet_id"].astype(str))
        before = len(test)
        test = test[test["customer_tweet_id"].astype(str).isin(golden_ids)].reset_index(drop=True)
        print(f"Filtered test set to golden-set rows only: {before} -> {len(test)}")

    os.makedirs("outputs", exist_ok=True)
    # gemini-3.5-flash-lite free tier: 15 requests/minute. Pace at ~4.5s/call to stay
    # comfortably under that (13.3/min) instead of bursting and hitting 429s.
    PACE_SECONDS = 4.5
    with open(C.AGENT_OUT, "w") as f:
        for i, row in test.iterrows():
            result = call_agent(row["customer_text"], retriever)
            result["customer_tweet_id"] = row["customer_tweet_id"]
            result["customer_text"] = row["customer_text"]
            f.write(json.dumps(result) + "\n")
            if i % 10 == 0:
                print(f"{i}/{len(test)} processed")
            time.sleep(PACE_SECONDS)
    print(f"Done -> {C.AGENT_OUT}")


if __name__ == "__main__":
    main()
