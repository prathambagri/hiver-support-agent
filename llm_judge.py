"""
LLM-as-judge: scores a (customer_message, reply) pair 1-5 on a fixed rubric.
Used by evaluate.py to score agent/baseline replies, and to build the
judge-vs-human agreement evidence the spec requires.
"""
import os
import json
import time
import google.generativeai as genai
import config as C

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(C.GEMINI_MODEL)

JUDGE_PROMPT = """You are grading a customer support reply. Score it 1-5 (integer) on
OVERALL quality, weighing:
- Relevance: does it actually address what the customer said?
- Grounding: does it avoid inventing policy/facts it can't support?
- Tone: is it appropriately empathetic and professional?

Customer message: {customer_text!r}
Proposed reply: {reply!r}

Return STRICT JSON only: {{"score": <1-5 integer>, "justification": "<one short sentence>"}}
"""


def judge_reply(customer_text: str, reply: str, retries: int = 4) -> dict:
    prompt = JUDGE_PROMPT.format(customer_text=customer_text, reply=reply)
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(prompt)
            raw = resp.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            parsed["score"] = int(parsed["score"])
            return parsed
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 20 if "ResourceExhausted" in type(e).__name__ or "429" in str(e) else 1.5
            print(f"  [retry {attempt+1}/{retries}] judge call failed: {type(e).__name__} - waiting {wait}s")
            time.sleep(wait)
    print(f"  [FALLBACK] judge giving up on this row: {last_err}")
    return {"score": None, "justification": f"judge_error: {last_err}"}
