# AmazonHelp AI Support Agent — Report

## 1. Problem framing: what "good" means here, and what I decided not to build

I picked AmazonHelp because it had by far the most volume in the dataset (15,323 tweets
vs. 2,600-5,500 for the next few brands), and the complaints clustered into a small,
intuitive set of buckets pretty quickly once I started reading through them — delivery
problems, refunds, billing weirdness, account access, wrong/broken items, and so on.

To me, "good" for this kind of agent isn't really about getting a single accuracy number
as high as possible. It's about three separate things that matter differently:

- **Getting intent right often enough that routing actually works.** If the agent
  misclassifies a refund request as a general inquiry, whoever picks it up next has to
  redo the triage work — the agent didn't help.
- **Drafting a reply that's actually usable**, meaning it's grounded in something real
  (a similar past resolution) rather than confidently making up a policy.
- **Escalating the right things.** This one matters the most to me, honestly. A support
  agent that never escalates isn't safe, and one that escalates everything isn't useful.
  The interesting engineering problem here is precision vs. recall on a rare, high-stakes
  class — get that wrong and you either overload humans with noise or let a real problem
  (fraud, account security, a customer about to churn) slip through untouched.

**What I chose not to build, and why:**
- No fine-tuning. I didn't have labelled training data going in — the whole point of this
  project was building that data (the golden set) from scratch, not training on it.
- No multi-language support. AmazonHelp gets tweets in German, French, Portuguese,
  Japanese, and more. I'm one annotator and don't speak most of those languages, so I
  restricted the test sample and golden set to English-only messages. This is a real
  scope decision, not laziness — a support agent that only works in English is genuinely
  less useful, and I've flagged it as the top thing I'd fix with more time.
- No full multi-turn thread context. Each customer message is handled as if it's the
  first message in the conversation, even though some of these are follow-ups in a
  longer back-and-forth. Reconstructing full threads reliably from the raw tweet graph
  turned out to be messier than I expected, and I didn't want to build something fragile
  under time pressure.
- One LLM call per message instead of three (classify, then draft, then decide) — mainly
  to keep cost and latency down. The tradeoff is that if the single call produces bad
  JSON, I lose all three outputs for that row instead of just one.

## 2. Results vs. baselines

I compared the agent against two baselines on the same 200-message golden set I hand-labelled:

| | Intent accuracy | Intent macro F1 | Escalate precision | Escalate recall | Escalate F1 | Judge score (1-5) |
|---|---|---|---|---|---|---|
| **My agent** | 0.755 | 0.746 | 0.912 | 0.517 | 0.660 | 4.22 |
| Simple baseline (keyword rules + nearest historical reply) | 0.610 | 0.467 | 0.000 | 0.000 | 0.000 | 2.15 |
| Trivial baseline (always same reply, never escalates) | 0.345 | 0.064 | 0.000 | 0.000 | 0.000 | 2.45 |

The gap between the agent and the trivial baseline isn't surprising — a system that always
guesses "general inquiry" and never escalates is basically a strawman. The more honest
comparison is against the simple baseline, since that one's actually doing real work
(keyword-based classification, retrieval-based reply). The agent still beats it clearly on
intent (0.755 vs 0.610 accuracy, and a much bigger macro F1 gap — 0.746 vs 0.467 — meaning
it's not just doing well on the common classes) and obviously on escalation, since the
rule-based baseline never escalates at all by design.

One thing that's genuinely interesting: the trivial baseline scored *higher* on the judge
metric (2.45) than the simple baseline (2.15), even though the simple baseline is a
"smarter" system. I think this is because the trivial baseline's generic canned reply
("thanks for reaching out, we'll follow up") is inoffensive and never actively wrong,
while the simple baseline sometimes serves up a retrieved past reply that doesn't
actually fit the new customer's situation, which reads as more clearly bad. That's a
small but real lesson: doing nothing risky can outscore doing something specific but
occasionally wrong, at least by an LLM judge's standards — which ties into the next
section.

## 3. Failure analysis: top 5 failure modes

1. **Escalation recall is the agent's biggest weakness — it misses about half the cases
   that should escalate (recall 0.517).** When I dug into the false negatives, most of
   them were messages with strong distress language but no explicit trigger word from my
   escalation rubric ("lawyer," "hacked," etc.) — for example, a customer who'd contacted
   support "again and again" without ever using the word "repeat." The rubric I wrote is
   too literal about specific phrases and doesn't generalize well to paraphrased distress.

2. **The LLM judge doesn't agree with my own judgment on reply quality.** I hand-scored
   40 of the agent's replies myself and compared to the judge's scores — the Spearman
   correlation was 0.069 (p=0.67), essentially no relationship. The judge gave a 5/5 to
   several replies that I scored 2-3 because they technically addressed the customer's
   message but ignored a second, more important complaint buried in the same tweet (e.g.
   a message about both a late delivery *and* a billing error, where the reply only
   handled the delivery part). The judge seems to reward surface-level politeness and
   grammatical correctness more than actually engaging with everything the customer said.

3. **Retrieval sometimes pulls a similar-sounding but substantively different past case.**
   Because retrieval is TF-IDF-based (word overlap, not meaning), a message about a
   *delayed* delivery can retrieve a past case about a *lost* delivery just because they
   share vocabulary like "package" and "arrived." The grounding then subtly steers the
   reply in the wrong direction.

4. **Non-English messages were excluded from evaluation entirely**, which means the
   headline numbers only tell you how the agent performs on English tweets — probably
   the majority of AmazonHelp's traffic, but not all of it. Performance on other
   languages is completely unmeasured and likely much worse, since the retrieval corpus
   and intent taxonomy were both built by reading English examples.

5. **The taxonomy's "general_inquiry" bucket is doing too much work.** Looking at the
   golden set, general_inquiry ended up as the single largest category (72 of 200 rows),
   which suggests it's functioning as a catch-all for anything that didn't cleanly fit
   the other seven categories rather than being a genuinely coherent intent. A few of
   those are probably actually distinct sub-intents I didn't think to define upfront
   (e.g. "how does this feature work" vs. "where's my order status").

## 4. What is misleading about my headline number

If I only reported "75.5% intent accuracy, 4.22/5 reply quality," that would paint a
rosier picture than what's actually going on, for a few reasons:

- **The 4.22/5 judge score is not trustworthy on its own** — see failure mode #2 above.
  Given essentially zero correlation with my own scoring, I don't think this number
  should be quoted without that caveat attached every time.
- **Escalation accuracy (0.840) looks good but is inflated by class imbalance.** Only 32%
  of the golden set actually needed escalation, so a system that leans toward
  *not* escalating scores well on raw accuracy even while missing real cases — which is
  exactly what's happening (recall of only 0.517). Precision/recall/F1 tell the real
  story; accuracy alone would have hidden this weakness completely.
- **The golden set is 200 messages from one brand, in one language, sampled once.** These
  numbers describe how the agent performs on AmazonHelp customer support tweets in
  English — they say nothing about how it'd generalize to a different brand, a different
  platform (email vs. Twitter), or non-English traffic.
- **The golden set itself was AI-assisted.** I had Claude propose initial labels for all
  200 rows based on the same rubric baked into the pipeline, then I reviewed and
  corrected them (with a specific focus on the escalate=TRUE rows, since those are the
  highest-stakes and most subjective calls). I'm confident in the final labels because I
  went through and adjusted them myself, but it's worth being upfront that the process
  wasn't 100% independent human labelling from a blank sheet — it was human review of
  AI-drafted suggestions, which is a meaningfully different (faster, but not identical)
  process than labelling from scratch.

## 5. What I'd do with one more week

- Fix the escalation rubric to catch paraphrased distress, not just specific trigger
  phrases — probably by giving the LLM a few worked examples of "this counts as repeat
  contact even though it doesn't say the word repeat" rather than a bullet-point rule.
- Try a better reply-quality judge, or at minimum a better-calibrated one — maybe by
  giving the judge a few of my own scored examples as few-shot calibration, or by asking
  it to score each part of a multi-part complaint separately instead of giving one
  holistic score.
- Swap TF-IDF retrieval for embeddings, to fix the "lost vs. delayed" retrieval confusion
  from failure mode #3.
- Extend the golden set to include a translated subset of non-English messages, even if
  just machine-translated, so I'd have *some* signal on cross-language performance instead
  of none.
- Split "general_inquiry" into 2-3 more specific sub-intents based on what's actually in
  that bucket, rather than leaving it as a catch-all.
- Reconstruct actual multi-turn threads instead of treating every message as
  conversation-opening, so the agent has real context when a customer is following up on
  something already discussed.

## 6. Decision log

See `DECISION_LOG.md` for the full list of 15 non-obvious decisions made throughout the
project, with reasoning for each.
