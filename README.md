# Refusal Cut 3.7B: research materials

This package documents a directional intervention in a 3.7B chat model. The most separated activation direction was not the best causal intervention: depth 34 produced zero clear attempts in selection, while depth 28 produced 22/30 and then 50/70 on the reserved float32 condition.

Read [the full tutorial](TUTORIAL.md), [measured results](RESULTS.md), and [record definitions](DATASET-CARD.md).

The public files include 1,150 annotation rows, aggregate results, the frozen selection record, source snapshots, and license/provenance notes. They contain no source prompt text, raw generated responses, private blinded packet, or model weights. Historical raw-ledger paths identify the private archive; their presence in prose does not mean those files are included.

## Model packages

Model repositories: [float32](https://huggingface.co/myronkoch/refusal-cut-3.7b) and [GGUF](https://huggingface.co/myronkoch/refusal-cut-3.7b-GGUF). Public [evaluation records](https://huggingface.co/datasets/myronkoch/refusal-cut-3.7b-evaluation). The models are independent derivatives of webAI-Official/TwIL-LM3-Pro, with no upstream endorsement. Model access and license terms are separate from the public record license.

## Reproduction status

The code directories are archival source snapshots. The historical continuation binds an original local checkpoint path; the evaluation helpers assume their original workspace layout. The complete fresh-machine preparation/extraction pipeline is not supplied. This package is useful for reviewing the method and saved results, but it must not be described as a tested one-command reproduction.

Original authored code and its documentation: MIT, see LICENSE-CODE.txt. Original records/annotations: CC BY-NC 4.0, see LICENSE-RECORDS.md. Upstream prompts, weights and third-party material retain their own terms. Method credit: Arditi et al. (2024), https://github.com/andyrdt/refusal_direction.
