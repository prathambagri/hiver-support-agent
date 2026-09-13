"""
Shared config. Edit these to tune sample sizes without touching pipeline code.
"""

# --- Brand choice ---
# AmazonHelp handle in the twcs.csv dataset. Change if you pick a different brand.
BRAND_HANDLE = "AmazonHelp"

# --- Paths ---
RAW_CSV = "data/twcs.csv"                     # raw Kaggle file (you download this)
THREADS_PARQUET = "data/threads.parquet"      # reconstructed (customer_msg, brand_reply) pairs for BRAND_HANDLE
CORPUS_PARQUET = "data/corpus.parquet"        # historical resolved pairs used for grounding/retrieval
TEST_PARQUET = "data/test_sample.parquet"     # held-out customer messages the agent runs on
GOLDEN_CSV = "data/golden_set.csv"            # hand-labelled ground truth
AGENT_OUT = "outputs/agent_results.jsonl"
BASELINE_TRIVIAL_OUT = "outputs/baseline_trivial_results.jsonl"
BASELINE_SIMPLE_OUT = "outputs/baseline_simple_results.jsonl"
JUDGE_SCORES_OUT = "outputs/judge_scores.csv"
HUMAN_JUDGE_SUBSET_OUT = "outputs/human_judge_subset.csv"

# --- Sizes (kept small deliberately so `run_all.py` finishes in <15 min on a laptop) ---
CORPUS_MAX_SIZE = 4000        # historical pairs available for retrieval grounding
TEST_SAMPLE_SIZE = 300        # messages the agent actually processes
GOLDEN_SET_SIZE = 200         # subset of TEST_SAMPLE_SIZE you hand-label (150-250 per spec)
JUDGE_AGREEMENT_SAMPLE = 30   # how many judge scores you also score yourself, to measure agreement
RANDOM_SEED = 42

# --- Intent taxonomy ---
# Defined FROM the data after eyeballing ~50 AmazonHelp customer messages. Adjust freely,
# but once you fix this list, keep it fixed for the rest of the pipeline + golden labelling.
INTENTS = [
    "delivery_or_shipping_issue",   # late/missing/lost package, tracking problems
    "refund_or_return",             # wants money back, return process questions
    "order_cancellation",           # wants to cancel an order
    "account_or_login_issue",       # can't log in, account locked, password reset
    "billing_or_payment_issue",     # wrong charge, double charge, payment failed
    "product_defect_or_wrong_item", # item broken, wrong item received, quality complaint
    "general_inquiry",              # status check, how-to questions, no clear complaint
    "compliment_or_other",          # praise, or doesn't fit any of the above
]

# --- Escalation policy (used by the agent's prompt AND the simple baseline's rule) ---
# Escalate when: refund/billing disputes over a implied dollar amount, account security issues,
# or clear strong negative sentiment (threats, legal language, repeated unresolved contact cues).
ESCALATION_GUIDANCE = """
Escalate to a human if ANY of these hold, otherwise auto-handle:
- The customer mentions legal action, fraud, a lawyer, or a chargeback/dispute.
- The issue is account security (hacked, unauthorized access, can't recover account).
- The customer explicitly says this is a repeat/unresolved contact ("again", "third time", "still not fixed").
- The message expresses strong distress/anger that a templated reply would clearly not defuse.
Otherwise handle automatically with a grounded reply.
"""

GEMINI_MODEL = "gemini-flash-lite-latest"  # free tier: 15 req/min (see agent.py pacing + DECISION_LOG.md #17)
