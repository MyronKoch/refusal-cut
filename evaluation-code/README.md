# Evaluation source snapshots

These are current source copies for review. The GGUF runner executed from the workspace `work/` directory; its saved run manifest matches the archived script hash. All 70 pairs are complete.

The approved Claude launcher authenticated but returned refusal with zero classifications. Its current source adds failed-stdout diagnostics; MANIFEST.json records the earlier executed source hash separately. Do not rerun it as an alternative successful rating. The successful second ledger was produced by a fresh gpt-6-astra subagent with no inherited conversation, given only the shuffled packet and fixed rubric.

The scripts currently derive paths from placement under the workspace `work/` directory. Copy these sources there to use the existing local layout. Do not execute them directly from this archive or describe them as a portable installer. Preserve model/input hashes and use a new run directory for changed settings.

`compare_blind_ratings.py` computes agreement after the second ledger is frozen, verifies mapping identity, and preserves original labels. `prepare_release_records.py` verifies ledger hashes and exports a strict field whitelist. Public exports contain 1,150 annotation rows and omit prompt/response text and reviewer rationale.

Publication copies replace personal path literals with placeholders. MANIFEST.json distinguishes the published hash from the pre-sanitization snapshot hash. Existing executed-source hashes remain historical evidence; sanitized runners have not been rerun.
