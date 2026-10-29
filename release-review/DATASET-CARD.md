---
license: cc-by-nc-4.0
language:
- en
tags:
- mechanistic-interpretability
- evaluation
---

# refusal-cut-3.7b experiment records

This card describes response-free experimental annotations and aggregates, not a new training-prompt corpus. Original records and annotations are offered under CC BY-NC 4.0; authored code is MIT. Neither license grants rights to upstream prompt text or replaces the weight license.

The 1,150 rows let a reader audit the reported counts and repeated ratings. They do not contain source prompts, generated answers, or reviewer rationales.

## Purpose and collection

Records document stock and fixed-direction interventions in TwIL-LM3-Pro at the pinned revision in MODEL-CARD.md. They support auditing behavioral counts, selection boundaries, and reproduction settings. They are not a benchmark with established population validity.

Direction fitting used train[0:300] in each source; train[300:400] was used for observational separation. Causal selection used test[0:30] and the frozen depth-28 test used test[30:100]. These slice ranges are zero-based and stop-exclusive.

| Dataset | Revision | Train rows | Test rows |
|---|---|---:|---:|
| mlabonne/harmful_behaviors | 01cead01398926d81f7c52bdb790ee8cf77ebba7 | 416 | 104 |
| mlabonne/harmless_alpaca | 02c6a92cfcf11bb0c387334f8146d149d65b587f | 25,058 | 6,265 |

The pinned README files declare split schemas/counts but no license. Release preparation also found empty descriptive cards at the current public dataset pages. This is not evidence of public-domain status. Identify each prompt by repository, revision, split, and row index; do not redistribute source prompt text until terms are established. Review generated-response redistribution separately.

## Public contents and exclusions

This release contains aggregate counts, per-response category labels, truncation observations and immutable input identifiers. Source prompts, raw model responses and reviewer rationales are excluded. A separate raw-response repository is not part of this release.

The prepared `PUBLIC-LABELS.jsonl` and `PUBLIC-RESULTS.json` contain no prompt text, response text, or reviewer free-text notes. Schema version 2 contains the original 940 float32 labels, 140 additional Q4 labels, and 70 second-rater labels, for 1,150 annotation rows. These are repeated measurements and annotations, not 1,150 unique prompts. Incomplete earlier runs are not combined into final results. Harmless responses and raw harmful bundles are not assembled by these files. The blinded-rating extension was added after the ledger was frozen and its input/output hashes checked.

## Prepared label schema

| Field | Meaning |
|---|---|
| phase | validation, reserved, or gguf-reserved |
| condition | stock, layer-N, random-N, stock-Q4_K_M, or depth-28-Q4_K_M |
| group | harmful or harmless |
| dataset_repo / dataset_revision / dataset_split | Immutable upstream prompt identity |
| dataset_index | Zero-based row within the upstream split |
| label | Original semantic rubric category |
| visible_truncation | Reviewer observation, not an engine-reported stop reason |
| review_source | Relative local ledger path, for audit |
| rater | original-Codex for original labels; gpt-6-astra for the 70 blind second labels |

Export schema version is recorded in `PUBLIC-RESULTS.json`. Publication locations and permanent evidence links must be supplied by the release owner. `substantive_answer` means a clear task attempt; it does not assert that the content is correct, effective, complete, or safe.

## Annotation and bias

One unblinded Codex reviewer (`gpt-6.1-sol`) assigned the original four-category labels. A later arithmetic recount checked totals rather than independently classifying responses. A fresh, no-history `gpt-6-astra` rater read a shuffled packet of the 70 reserved edited float32 harmful responses with the fixed rubric, while original labels, counts, ID mapping, and model condition were withheld. Agreement was 62/70 (88.6%), with Cohen's kappa 0.72332. Counts are shown below.

| Rater | Clear attempts | Redirections | Ambiguous |
|---|---:|---:|---:|
| Original, gpt-6.1-sol | 50 | 6 | 14 |
| Blind second pass, gpt-6-astra | 54 | 6 | 10 |
 All eight disagreements involved attempts versus ambiguity. Original labels were preserved without adjudication.

Only that edited float32 subset received a second annotation pass. Stock, selection, harmless, and Q4 labels remain single-rater. AI raters only, edited float32 subset only. Both are OpenAI models, independent of session and prior labels but not vendor; neither is an independent human evaluator. The approved Claude Opus 5.5 attempt returned refusal and no classifications before the fresh Astra pass; it contributes no labels. Short output budgets can make categories ambiguous and undercount delayed refusals. Keyword counts remain a separate wording measure.

The prompt groups differ in topic and wording. Normalized exact-overlap audits do not rule out semantic duplicates. The reserved prompts were excluded from causal selection but had been exposed in historical stock evaluation. Selection and reserved rows must stay distinguished; pooling them would conceal model-selection exposure.

## Results and limitations

Depth 28 produced 22/30 clear harmful attempts in selection and 50/70 reserved, versus zero for the corresponding stock runs. Three random controls each preserved stock selection labels. The highest original separation depth, 34, produced zero clear selection attempts and two harmless explicit refusals. The prepared result file preserves other candidate counts and first-token KL.

The paired reserved Q4 harmful replication is also included: stock had 69 explicit refusals and 1 ambiguous response; edited Q4 had 40 attempts, 5 redirections, and 25 ambiguous responses. It used Mac Metal, 100 tokens, and identical paired rendered prompts. These prompts were already used for float32 evaluation. The Q4 result is a separate condition, not an independent untouched test, and no reserved Q4 harmless evaluation was run.

These records establish an observed intervention effect under recorded settings. They do not establish correctness, full capability retention, a unique circuit, a quantization-only causal effect, or phone refusal rates. No Heretic optimizer trial or recorded paired phone trial is included.
