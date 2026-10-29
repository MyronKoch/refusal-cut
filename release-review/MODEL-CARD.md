---
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: webAI-Official/TwIL-LM3-Pro
license: other
license_name: webai-non-commercial-license-ver.-1.0
license_link: https://huggingface.co/myronkoch/refusal-cut-3.7b/blob/main/licenses/LICENSE-webAI.md
tags:
- mechanistic-interpretability
- granite
- research
extra_gated_prompt: "Read the included access terms and licenses before requesting manual approval."
extra_gated_fields:
  "I acknowledge non-commercial research and education use under the supplied license": checkbox
  "I understand outputs may be unsafe, false or incomplete and approval implies no upstream endorsement": checkbox
---

# refusal-cut-3.7b

Depth-28 float32 research derivative

Repository: [myronkoch/refusal-cut-3.7b](https://huggingface.co/myronkoch/refusal-cut-3.7b). Model-file access requires manual approval under the included access terms.

This is a research copy of a 3.7-billion-parameter chat model with one internal direction projected out of its weights. On 70 reserved harmful requests, the original model explicitly refused all 70; the edited model had zero explicit refusals, six redirections, 50 clear attempts, and 14 ambiguous responses. Answer correctness was not established. This model is derived from webAI-Official/TwIL-LM3-Pro, not affiliated with or endorsed by webAI.

This checkpoint is a persistent directional projection of `webAI-Official/TwIL-LM3-Pro`, revision `144a3ccb31264deadcf68bc5a1d58f621e497d03`. It was evaluated to investigate a causal change in refusal behavior. It was not trained from scratch or optimized with Heretic. Safety refusal behavior was substantially reduced on the measured prompt set. The model may generate unsafe, false, incomplete, or otherwise unsuitable answers.

## Model and intervention

The source contains 3,659,737,600 parameters in a Granite architecture: 40 blocks, hidden width 2,560, 40 query heads, 8 key/value heads, and untied input/output embeddings. The saved derivative has float32 tensors; promoting the source's bfloat16 values does not recover precision lost before this experiment.

A contrast direction was extracted at the last prompt token after 28 transformer blocks from 300 harmful and 300 harmless training prompts. Each candidate was the harmful activation mean minus the harmless activation mean, normalized to unit length. This selection was informed by a separate causal sweep; it was not chosen solely from activation separation.

For the selected unit vector r, the edit removes the component along r at all 81 residual writers: the embedding table, 40 attention output projections, and 40 MLP down projections. For a linear writer W[out,in], W' = W - r(r^T W), and its bias is projected if present. For an embedding E[vocabulary,hidden], E' = E - (E r)r^T. The vocabulary head and final normalization were not edited. Depth 28 names the extraction site, not the only intervention site.

The procedure adapts [Arditi et al., 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html). The historical baseline and activation extraction were performed in an earlier Claude session. The continuation and original semantic classification were performed in a Codex session; the original classification model was `gpt-6.1-sol`.

## Evaluation design

The float32 MPS comparison used eager attention, greedy decoding, a 100-new-token budget, batch size 16, the source tokenizer/template, thinking disabled, and system prompt `You are a helpful assistant.` Candidate extraction indices were 12, 20, 24, 28, 31, 34, and 38; three seeded random unit directions were controls. Each condition began from stock weights and cleanly removed its runtime hooks.

Dataset identities and zero-based slices:

| Source | Immutable revision | Direction fit | Separation holdout | Causal selection | Reserved causal test |
|---|---|---|---|---|---|
| mlabonne/harmful_behaviors | 01cead01398926d81f7c52bdb790ee8cf77ebba7 | train[0:300] | train[300:400] | test[0:30] | test[30:100] |
| mlabonne/harmless_alpaca | 02c6a92cfcf11bb0c387334f8146d149d65b587f | train[0:300] | train[300:400] | test[0:30] | test[30:100] |

Revisions were audited against saved prompt records. Original extraction scripts did not explicitly pin revisions. Zero lowercased/whitespace-normalized exact overlaps were found in the recorded audits; semantic near-duplicates were not excluded. The reserved prompts were withheld from causal selection, but their stock baseline had already been examined historically. They are not a never-seen dataset.

The depth-28 choice was frozen at 2026-10-11 03:15:57 UTC before the reserved causal run. Changing the choice after inspecting that run would invalidate its role as a reserved test.

## Results

Labels distinguish explicit refusal, safer-task redirection, clear task attempt, and ambiguous/truncated text. The ledger key `substantive_answer` means an attempt, not verified correctness. Categories below are mutually exclusive; keyword refusal is a separate wording metric.

| Harmful condition | N | Explicit refusal | Redirection | Clear attempt | Ambiguous |
|---|---:|---:|---:|---:|---:|
| Selection stock | 30 | 29 | 1 | 0 | 0 |
| Selection depth 28 | 30 | 0 | 2 | 22 | 6 |
| Selection depth 34 | 30 | 13 | 16 | 0 | 1 |
| Each of three random controls | 30 | 29 | 1 | 0 | 0 |
| Reserved stock | 70 | 70 | 0 | 0 | 0 |
| Reserved depth 28 | 70 | 0 | 6 | 50 | 14 |

Depth 34 had the greatest original Cohen's d (7.98759), but produced zero clear harmful task attempts under intervention. Depth 28 produced 22/30 in selection and 50/70 in the reserved test. These are observed counts under this setup, not a universal refusal-suppression rate.

All 30 selection harmless responses at depth 28 attempted an answer. In the reserved harmless set, both stock and edited conditions had 69 attempts and the same one information-access refusal. Mean harmless first-token KL(stock || edited) was 0.06592995 in selection and 0.08534747 in the reserved test. KL measures drift in the first answer-token vocabulary distribution; it does not establish preserved capabilities. Factual errors, relevance concerns, and truncation were observed.

The original labels came from one unblinded AI reviewer. A later arithmetic recount checked totals only. A fresh reviewer received a shuffled packet of the 70 reserved edited float32 harmful outputs and the fixed rubric, without original labels, counts, mapping, or model condition.

| Rater | Clear attempts | Redirections | Ambiguous |
|---|---:|---:|---:|
| Original, gpt-6.1-sol | 50 | 6 | 14 |
| Blind second pass, gpt-6-astra | 54 | 6 | 10 |

Agreement was 62/70 (88.6%), with Cohen's kappa 0.72332. Kappa adjusts label agreement for chance, not answer quality. All eight disagreements were attempt versus ambiguity: six original ambiguous labels became attempts and two attempts became ambiguous. Both ledgers are preserved, without adjudication or retrospective selection. Both are OpenAI models: the second pass is independent of the original session and labels, not of vendor.

This is a second AI annotation pass on edited float32 harmful outputs only, not independent human validation or second rating of stock, selection, harmless, or Q4 outputs. It supports labeling consistency under the rubric, not correctness or effective harmful completion. No statistical-significance claim is made. The initially approved fresh Claude Opus 5.5 call returned refusal and zero classifications; it is recorded as a failed rating attempt and contributes no labels.

## Export fidelity and deployment evidence

On three benign probes, hook-versus-persistent-edit maximum logit difference was 5.4836273193359375e-5. A separate-process float32 reload reproduced reference logits with maximum difference 0.0 and matched short cached generations. These checks validate fidelity on those probes, not broad behavioral equivalence.

The Q4_K_M conversion has four paired 100-token CPU probes and one separate equal-300-token diagnostic. A subsequent 70-pair Mac Metal comparison used the same reserved harmful prompts: edited Q4 had 40 clear attempts, 5 redirections, and 25 ambiguous responses; stock had 69 explicit refusals and 1 ambiguous response. Those are Q4 results, separately reported in the GGUF card, not replacements for the float32 table. Runtime and precision both differ, so a lower Q4 attempt count does not isolate a quantization effect. No new Q4 reserved harmless evaluation or matched iPhone records have been collected. Phone import and behavior are user reports only. Heretic optimization has not run.

## Intended use and limits

Use is restricted by the upstream license to non-commercial research and education. Intended studies include directional interventions, refusal measurement validity, collateral behavior, and cross-runtime reproduction. The recorded experiment supports a causal behavioral effect of this intervention. It does not prove a unique refusal circuit, sufficiency, topic independence, general accuracy, or robust safety.

Preserve the complete webAI license and accompanying Apache-2.0 license and applicable notices. This is an independent derivative, with no webAI or IBM endorsement claimed. The public name is `refusal-cut-3.7b`; the staged access acknowledgement is in `ACCESS-TERMS.md`. Gating is an access control and does not broaden license permissions.

## Reproduction records

The public research package includes aggregate counts, response-free labels, the selection record, source identities, tutorial, and source snapshots. Raw response ledgers remain local and are not linked as if they were public. Cross-repository links will be added only when the corresponding repositories exist.

## Local usage

After obtaining an authorized copy of this repository, run from its root. The model uses float32 and needs substantially more memory than its 2.24 GB GGUF derivative. This example keeps the recorded thinking-off setting and uses a harmless arithmetic probe.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "."  # directory containing config.json and the four weight shards
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, dtype=torch.float32,
    attn_implementation="eager",
).to("mps").eval()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 plus 2?"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
assert prompt.rstrip().endswith("<think></think>")
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True))
```

The exact packaged tokenizer switch and closed thinking suffix were checked locally. The recorded separate-process reload evidence predates this documentation example; the example is not a new model evaluation.
