# License and source audit

Checked 2026-10-11. This is a packaging evidence record; final distribution choices remain with the release owner.

## Exact model revision

The model API returned checkpoint SHA `144a3ccb31264deadcf68bc5a1d58f621e497d03` and listed `LICENSE.md` and `apache-2.0-LICENSE.txt`. Both files were retrieved from that immutable revision and preserved under `licenses/`. LICENSE-SOURCES.json records retrieval URLs and local hashes.

- [Pinned webAI license](https://huggingface.co/webAI-Official/TwIL-LM3-Pro/blob/144a3ccb31264deadcf68bc5a1d58f621e497d03/LICENSE.md)
- [Pinned accompanying Apache-2.0 file](https://huggingface.co/webAI-Official/TwIL-LM3-Pro/blob/144a3ccb31264deadcf68bc5a1d58f621e497d03/apache-2.0-LICENSE.txt)

webAI sections 2.1 and 3.1 permit derivative distribution subject to complete license inclusion and retention of notices. Section 3.2 requires the non-commercial restriction to remain on derivatives; section 3.3 limits use to non-commercial research and education. Section 3.5 restricts use of licensor marks and endorsement implications. It contains no required derivative naming formula. This does not automatically clear the proposed branded filename: choose a public name with clear independent attribution and no endorsement claim.

Include both complete license files and review copyright/attribution notices before final packaging. Do not label the derivative Apache-2.0-only. Gating and an acknowledgement of reduced refusal behavior are separate access conditions; neither expands upstream permissions. Code and experimental annotations need their own reviewed license choices rather than inheriting the weight license indiscriminately.

## Dataset revisions

- [harmful_behaviors pinned card](https://huggingface.co/datasets/mlabonne/harmful_behaviors/blob/01cead01398926d81f7c52bdb790ee8cf77ebba7/README.md)
- [harmless_alpaca pinned card](https://huggingface.co/datasets/mlabonne/harmless_alpaca/blob/02c6a92cfcf11bb0c387334f8146d149d65b587f/README.md)

Both pinned README files contain dataset schema and split counts but no license declaration or descriptive licensing terms. Their exact retrieved content is preserved. The current public cards also have no descriptive content. This check does not establish that no license exists elsewhere or through their source datasets. Prompt redistribution permission is unresolved, not permissive by default.

Until resolved, publish prompt identities and indices rather than prompt text. Review harmless and harmful generated-response redistribution separately, including any copied source text. A gate does not cure missing copyright permission. Public notes must be reviewed for embedded prompts/completion excerpts.

## Remaining before an upload

1. Retain applicable copyright/attribution notices from upstream alongside the preserved licenses.
2. Use `refusal-cut-3.7b` and `refusal-cut-3.7b-GGUF`, with clear upstream attribution and no endorsement claim. Add live repository links only after creation.
3. Establish dataset provenance/terms or keep source prompts unredistributed.
4. Authored code is MIT; original annotations and records are CC BY-NC 4.0. Third-party material and source prompts retain their own terms. The derivative weights remain under the applicable upstream licenses.
5. Verify the actual repository gate and staged file list, then obtain the user's upload approval.
