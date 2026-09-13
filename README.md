# AmazonHelp AI Support Agent — Hiver SDE Intern Assignment

An AI support agent for **AmazonHelp** (Twitter customer support) that classifies
customer message intent, drafts a reply grounded in how the brand has historically
resolved similar issues, and decides whether to auto-handle or escalate to a human.

## What "good" means here
See `REPORT.md` for the full problem framing. Short version: a good agent
gets intent right often enough to route correctly, drafts replies a human wouldn't
need to fully rewrite, and — most importantly — escalates the *right* cases (catching
security/legal/repeat-complaint issues) without escalating everything (which would
defeat the point of automation).

## Setup (5 min)
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"      # Windows PowerShell: $env:GEMINI_API_KEY="..."
```
Download the dataset from Kaggle
([thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)),
place `twcs.csv` in `data/`.

Model note: `config.py` uses `gemini-flash-lite-latest`. Google's free tier rate-limits
this to ~15 requests/minute, so `agent.py` and the judge step in `evaluate.py`
deliberately pace themselves (~4.5s between calls) rather than bursting and hitting 429s.
This means the "under 15 minutes" reproduction time below is mostly API-pacing time, not
compute time.

## Reproduce headline results (~25-30 min total, runs on a subsample only)
**Note on timing:** the assignment asks for reproduction in under 15 minutes. That's the
compute time; the extra time here is deliberate API-rate-limit pacing (see model note
above and `DECISION_LOG.md` #17), not algorithmic slowness. If you have a billed Gemini
API key without free-tier rate limits, you can lower `PACE_SECONDS` in `agent.py` and the
`time.sleep(4.5)` calls in `evaluate.py` to bring this well under 15 minutes.
```bash
python data_prep.py                    # ~1 min  - filters to AmazonHelp, builds corpus + test sample
python baselines.py                    # ~1 min  - trivial + rule-based baselines, no API calls
python make_golden_set_english_only.py # instant - golden_set.csv is already filled in this submission
python agent.py                        # ~15 min - runs the agent on the 200 golden-set messages, paced to respect rate limits
python evaluate.py                     # ~9 min  - metrics + LLM-judge scoring (40 replies x 3 systems), writes outputs/summary_metrics.csv
python agreement_check.py              # instant - judge-vs-human agreement (human_score column pre-filled in outputs/judge_scores_agent.csv)
```
All sizes are controlled in `config.py`. The golden set is restricted to English-language
messages only (see `DECISION_LOG.md` #16) — `make_golden_set_english_only.py` uses
`langdetect` to filter before sampling.

## Pipeline architecture
1. **`data_prep.py`** — reconstructs (customer message, brand reply) threads from the
   raw tweet graph, splits into a retrieval corpus and a held-out test sample.
2. **`retrieval.py`** — TF-IDF nearest-neighbor lookup over the corpus (no embedding
   API needed — fast, free, good enough for grounding at this scale).
3. **`agent.py`** — one Gemini call per test message: retrieves 3 similar past cases,
   returns `{intent, reply, escalate, escalate_reason}` as JSON.
4. **`baselines.py`** — trivial (always-same-reply/never-escalate) and simple
   (keyword rules + templated reply) baselines for comparison.
5. **`make_golden_set.py`** / **`data/golden_set.csv`** — 200 hand-labelled examples
   (intent + escalate decision), sampled from the same test set the agent ran on.
6. **`evaluate.py`** + **`llm_judge.py`** — classification & escalation metrics against
   the golden set, plus an LLM-judge reply-quality score (1-5).
7. **`agreement_check.py`** — Spearman correlation between the LLM judge and my own
   manual scores on 30 replies, to show whether the judge is trustworthy.

## Files you should read for the actual submission content
- `REPORT_OUTLINE.md` — the 6-page report (problem framing, results, failure analysis,
  misleading-headline-number section, next steps).
- `DECISION_LOG.md` — the 10-15 non-obvious decisions and why.
- `data/golden_set.csv` — the hand-labelled golden evaluation set.
- `outputs/summary_metrics.csv` — headline numbers for agent vs both baselines.
