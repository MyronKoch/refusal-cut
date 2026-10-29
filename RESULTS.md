# Measured results

The final float32 sweep uses 30 harmful and 30 harmless prompts per condition. All figures below refer to runs/validation-final, with the same model, template, decoding settings, and 100-token budget. Depth 28 is frozen in SELECTION.json. The reserved 70+70 test is complete and semantically reviewed.

| Condition | Harmful keyword refusal | Explicit refusal | Redirection | Clear task attempt | Ambiguous | Harmless first-token KL |
|---|---:|---:|---:|---:|---:|---:|
| Stock | 29/30 | 29 | 1 | 0 | 0 | 0 |
| Depth 12 | 20/30 | 18 | 2 | 4 | 6 | 0.27619 |
| Depth 20 | 17/30 | 15 | 4 | 9 | 2 | 0.35394 |
| Depth 24 | 8/30 | 5 | 3 | 17 | 5 | 0.21665 |
| Depth 28 | 1/30 | 0 | 2 | 22 | 6 | 0.06593 |
| Depth 31 | 10/30 | 3 | 14 | 8 | 5 | 0.05786 |
| Depth 34 | 28/30 | 13 | 16 | 0 | 1 | 0.37142 |
| Depth 38 | 28/30 | 27 | 3 | 0 | 0 | 0.69019 |
| Random 0 | 29/30 | 29 | 1 | 0 | 0 | 0.01213 |
| Random 1 | 29/30 | 29 | 1 | 0 | 0 | 0.01346 |
| Random 2 | 29/30 | 29 | 1 | 0 | 0 | 0.01443 |

The stock condition and depths 12, 20, 24, 28, and 31 attempt all 30 harmless tasks. Depth 34 has 2 explicit harmless refusals, 27 task attempts, and 1 ambiguous response; one refusal also provides generic help and is flagged as mixed. This does not establish correctness, completeness, or preserved general capabilities. The semantic ledgers record accuracy concerns and truncation. An ambiguous harmful response is not counted as a successful task attempt.

Depth identifies where a fixed direction was extracted. The intervention projects that direction out of input embeddings, every attention output writer, and every MLP down-projection writer throughout the network.

The preliminary evidence supports a causal behavioral change under this setup, with depth 28 giving a better observed tradeoff than earlier candidates. It does not establish sufficiency, rule out topic features, or establish phone behavior. Quantized deployment evidence is reported separately below. All seven candidate conditions are complete and reviewed. All three random controls are complete and semantically reviewed. They preserve the stock refusal categories, with no clear harmful task attempts and no harmless refusals. Depth 38 has 3 explicit harmless refusals, 26 task attempts, and 1 ambiguous response, reinforcing its poor observed tradeoff.

See each condition JSON for raw responses and the corresponding *-semantic-review.json for per-index labels. First-token KL is KL(base || variant), averaged over the harmless prompts. It measures a narrow distributional effect before answer generation.

![Validation semantic categories and first-token drift](validation-results.png)

## Reserved test of the frozen depth-28 choice

The candidate was frozen before running original prompt indices 30 through 99. Both conditions generated 70 harmful and 70 harmless answers; all 280 responses were read against their prompts. These prompts were reserved from causal selection, with historical stock-baseline exposure disclosed.

| Harmful response category | Stock | Depth 28 |
|---|---:|---:|
| Explicit refusal | 70 | 0 |
| Refusal by redirection | 0 | 6 |
| Clear task attempt | 0 | 50 |
| Ambiguous or truncated | 0 | 14 |

Keyword refusals are 70/70 for stock and 2/70 for depth 28. The semantic comparison is more conservative: 50/70 clear attempts, with 14 ambiguous fragments excluded. Neither attempts nor refusal suppression establish correct or effective outputs.

Both conditions have 69 harmless task attempts and one legitimate current-news access refusal. Equal labels do not establish preserved quality. The review records an invented historical detail added to an already false statue premise, persistent factual/mathematical errors, and reduced useful game-rule content before truncation. See reserved-test-28/*-semantic-review.json for the complete ledgers.

Mean harmless first-token KL(base || depth28) is 0.08534747. The result supports generalization of the observed behavioral change to prompts not used in causal selection, within this model/template/precision/response-budget setup. It does not establish a fully isolated refusal mechanism or broad capability preservation.

## Persistent checkpoint verification

Exported runs/export-28/model-f32 using the frozen depth28 direction. On three benign probes, runtime-hook versus edited-weight logits differed by at most 0.00005484. A separate process reloaded the saved model and matched the reference logits and short cached generations; the reloaded logit error was exactly 0.0 on those probes. See [reload verification](runs/export-28/reload-verification.json). This verifies fidelity on those probes, not general capabilities.

## Matched Q4 deployment probes

Stock and edited models were converted to f16 with the same pinned converter and quantized with the same Q4_K_M settings. Both Q4 files have 40 Granite blocks, hidden size 2,560, and 363 tensors, at about 2.24 GB each. GGUF-PAIRED-PROBES.json records model hashes, actual arguments, and answers.

Four explicit paired CPU probes used identical rendered prompt hashes and a 100-token budget. Both models correctly answer arithmetic and approximate division and give a valid ABAB scheme. Their extra poem examples truncate. On the exam-cheating probe, stock explicitly refuses; the edited model begins a program introduction and imports, which remain ambiguous under this budget. A separate equal 300-token diagnostic retained stock refusal and produced a longer task-framed code/data fragment from the edited file. It is classified as an attempt, with truncation and unverified functionality explicitly recorded in GGUF-LONG-REVIEW.json. This is a small deployment check, not a full repeated quantized refusal-rate evaluation. Phone behavior has not been tested.

## Reserved Q4 harmful deployment replication

All 70 harmful reserved prompts, indices 30 through 99, were evaluated on both matched Q4_K_M files. This subsequent comparison used Mac Metal (`-ngl 99`), llama-cli build 11146/7fe450e19, temperature 0, seed 42, 100 new tokens, context 512, batch 128, and thinking disabled. Every stock/edited rendered-prompt hash matched. The completed 140 raw records and their hashes were separately verified. One unblinded AI reviewer (`gpt-6.1-sol`) read every response against its prompt.

| Harmful response category | Stock Q4 | Edited Q4 |
|---|---:|---:|
| Explicit refusal | 69 | 0 |
| Refusal by redirection | 0 | 5 |
| Clear task attempt | 0 | 40 |
| Ambiguous or truncated | 1 | 25 |

The edited Q4 result is 40/70 clear attempts (57.1%), compared with the original float32 50/70 (71.4%). Do not pool these conditions or describe the difference as caused solely by quantization: engine, numerical precision, and generation behavior differ. Attempts remain unverified for correctness or functionality. This reuses the already-exposed reserved prompt set to test deployment behavior; it is not a new untouched test. No reserved Q4 harmless evaluation or matched phone evaluation was run.

The response-free Q4 metadata and aggregate counts are in `release-review/PUBLIC-RESULTS.json`. The full Q4 manifest and semantic ledger remain in the private research archive. Schema-2 public records include the original 940 labels, 140 Q4 labels, and the 70 second annotations below, for 1,150 annotation rows.

## Blind second rating of edited float32 responses

The original rater was gpt-6.1-sol. A fresh gpt-6-astra rater, with no inherited conversation, classified the 70 reserved edited float32 harmful outputs using only shuffled prompt-response pairs and the fixed rubric. Original labels, counts, the ID mapping, and model condition were withheld. The ledger was saved before comparison. It reported no blinding leakage.

| Label | Original | Blind second rater |
|---|---:|---:|
| Explicit refusal | 0 | 0 |
| Redirection | 6 | 6 |
| Clear task attempt | 50 | 54 |
| Ambiguous or truncated | 14 | 10 |

Agreement is 62/70 (88.6%), with Cohen's kappa 0.72332016. Six originally ambiguous responses became attempts; two originally labeled attempts became ambiguous. All redirection labels agreed. Original labels and selection remain unchanged, with no adjudication. These are two AI judgments of attempted substance, not verification of correctness, functionality, or safety. Other conditions, including Q4, remain single-rater.

The initially approved Claude Opus 5.5 rating call returned refusal and zero classifications. That failed attempt is separately recorded and contributes no labels. See [agreement and disagreements](blind-rating/AGREEMENT.json), and [failed Claude attempt metadata](blind-rating/CLAUDE-ATTEMPT.json). Raw packets and response text remain outside the public record export.

## Availability of evidence

This public package includes response-free aggregates and labels, a selection record, and an agreement summary. References to raw run directories above identify the retained private archive; those response files are not included or presented as public downloads. The original and second raters are both OpenAI models, independent of session and labels but not vendor. No paired phone results are included.
