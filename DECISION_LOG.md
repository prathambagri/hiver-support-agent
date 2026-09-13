# Decision Log

1. **Brand: AmazonHelp.** High tweet volume in the dataset and issues cluster into a
   small number of intuitive categories (delivery, refund, billing, account), which
   made defining a fixed intent taxonomy tractable in the time available.

2. **Fixed, hand-defined intent taxonomy (8 classes) instead of unsupervised
   clustering.** Clustering on ~3M noisy tweets would need real tuning time I didn't
   have; 8 classes derived from skimming ~50 real messages is defensible and keeps
   labelling/evaluation simple.

3. **TF-IDF retrieval instead of embeddings.** No API latency/cost, instant to build,
   and at corpus size ~4,000 it retrieves near-duplicate past complaints well enough
   for grounding. Embeddings would likely help recall on paraphrased complaints —
   noted as future work.

4. **Corpus/test split is random, not time-based.** A time-based split (train on
   older tweets, test on newer) would better simulate production, but random split
   was faster to implement and the dataset's policies likely didn't shift much within
   its collection window. Risk: some optimistic leakage if a customer messaged twice.

5. **Single LLM call per message for intent+reply+escalate, not three separate
   calls.** Cuts API cost/latency 3x. Risk: one bad JSON parse loses all three
   outputs for that row — mitigated with retries and a fail-safe fallback (see
   `agent.py::call_agent`).

6. **Escalation policy is a fixed rubric given to the LLM (see `config.py`), not
   learned.** No labelled training data for escalation existed before this project;
   a hand-written rubric is inspectable and lets the LLM apply judgment beyond exact
   keyword matches (unlike the simple baseline, which IS pure keyword matching).

7. **Simple baseline reply = nearest historical reply verbatim (k=1 retrieval), not
   a hand-written template per intent.** Writing 8 good templates would take real
   time; using the single nearest historical reply is a fair "retrieval-only, no
   generation" baseline that isolates the value the LLM step adds.

8. **Golden set sampled from the test set, not the full corpus.** Guarantees every
   golden row has a corresponding agent/baseline prediction to compare against
   without a separate reconciliation step.

9. **Golden set size: 200** (within the 150-250 spec range). Chosen at the middle
   rather than the max to keep hand-labelling time bounded given the deadline.

10. **Intent ground truth judged from customer message alone, not the dataset's own
    brand reply.** Using the actual historical reply as a shortcut to infer intent
    would bias toward whatever AmazonHelp's own agents historically believed the
    intent was, not what it actually is.

11. **LLM-as-judge scores reply quality only (not intent or escalation) —** those two
    have unambiguous ground truth already; using an LLM judge for them would just add
    noise where a hard metric exists.

12. **Judge-human agreement measured with Spearman correlation on a 30-example
    subset, not exact-match accuracy.** Reply quality is graded on an ordinal 1-5
    scale where "off by one point" matters less than rank agreement — correlation is
    the more honest metric for that.

13. **All three systems (agent + 2 baselines) are judge-scored on the exact same
    golden-set rows.** Otherwise a difference in mean judge score could be an
    artifact of which rows got sampled rather than a real quality gap.

14. **Escalation is reported via precision/recall/F1 for the "escalate" class, not
    just accuracy.** Escalation is rare in the data (imbalanced), so raw accuracy
    would look artificially high for any system that just never escalates — see
    "What's misleading about my headline number" in the report.

15. **No fine-tuning, no full-thread (multi-turn) context — deliberately out of scope.**
    Explained in `REPORT.md` under "what I chose not to build."

16. **Restricted the golden set (and therefore the reported headline numbers) to
    English-language messages only.** AmazonHelp gets tweets in German, French,
    Portuguese, Japanese and more, and I'm a single non-multilingual annotator — I
    can't reliably hand-label intent/escalation in a language I don't read. This is a
    real scope limitation, documented in the report, not something I tried to hide.

17. **Used `gemini-flash-lite-latest` instead of the originally-planned model.**
    `gemini-2.0-flash` returned a hard 404 (deprecated) partway through the build, and
    the next model I tried had a 20-requests/*day* free-tier cap that made a full run
    impossible. Switched to a lite-tier model with a 15-requests/*minute* limit instead,
    and added explicit pacing (~4.5s between calls) to stay under it rather than burning
    retries against a wall.

18. **Golden set labelling was AI-assisted, not done from a blank sheet.** I had Claude
    (used as an AI coding/analysis assistant, consistent with the assignment's stated
    rules) propose an initial intent + escalate label for all 200 rows based on the same
    rubric already baked into the pipeline's prompt. I then reviewed and corrected all of
    them myself, focusing extra attention on the escalate=TRUE rows since those are the
    highest-stakes, most judgment-heavy calls. This is disclosed explicitly in the report
    (section 4) since it's a meaningfully different process from independent labelling
    and I'd rather be upfront about it than have it come up as a surprise live.

## AI assistance disclosure
This project's pipeline code (`agent.py`, `retrieval.py`, `evaluate.py`, `llm_judge.py`,
`baselines.py`, `data_prep.py`, and supporting scripts) was written with Claude (Anthropic)
as a coding assistant, per the assignment's explicit allowance for AI coding tools. I
directed the design decisions (see entries above), debugged real issues that came up
during the actual run (a deprecated model, then a rate-limit wall — see #17), and can
walk through and modify any part of this code live. The golden-set labels were
AI-suggested and human-reviewed, as described in #18 above and in `REPORT.md` section 4.
