"""Prepare response-free review records from existing, final experiment ledgers."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs/refusal-cut"
OUT = ROOT / "release-review"
LABELS = {"explicit_refusal", "refusal_by_redirection", "substantive_answer", "ambiguous_or_truncated"}
DATASETS = {
    "harmful": ("mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7"),
    "harmless": ("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f"),
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

records, summaries = [], []
for phase, folder, offset, expected in [
    ("validation", "validation-final", 0, 30), ("reserved", "reserved-test-28", 30, 70)
]:
    for ledger in sorted((ROOT / "runs" / folder).glob("*-semantic-review.json")):
        review = json.loads(ledger.read_text())
        name = ledger.name.removesuffix("-semantic-review.json")
        raw = json.loads(ledger.with_name(name + ".json").read_text())
        counts = {}
        for group, (repo, revision) in DATASETS.items():
            rows = review.get(group + "_labels", review.get(group))
            assert isinstance(rows, list) and len(rows) == expected, ledger
            assert len(raw[group + "_responses"]) == expected, ledger
            indexes = []
            for row in rows:
                index = row.get("original_index", offset + row["index"])
                indexes.append(index)
                assert row["label"] in LABELS, row["label"]
                records.append({
                    "phase": phase, "condition": "stock" if name == "baseline" else name,
                    "group": group, "dataset_repo": repo, "dataset_revision": revision,
                    "dataset_split": "test", "dataset_index": index, "label": row["label"],
                    "visible_truncation": row.get("visible_truncation"),
                    "review_source": str(ledger.relative_to(ROOT)),
                    "rater": "original-Codex",
                })
            assert sorted(indexes) == list(range(offset, offset + expected)), ledger
            counts[group] = dict(Counter(row["label"] for row in rows))
            stored = review["counts"][group]
            for label in LABELS:
                assert counts[group].get(label, 0) == stored.get(label, 0), (ledger, group, label)
        summaries.append({
            "phase": phase, "condition": "stock" if name == "baseline" else name,
            "n_per_group": expected, "semantic_counts": counts,
            "harmful_keyword_refusals": raw["harmful_keyword_refusals"],
            "harmless_keyword_refusals": raw["harmless_keyword_refusals"],
            "first_token_kl_base_to_variant_mean": raw["first_token_kl_base_to_variant_mean"],
            "review_source": str(ledger.relative_to(ROOT)), "review_sha256": digest(ledger),
        })
assert len(summaries) == 13 and len(records) == 940
extensions = {}
agreement_path = ROOT / 'blind-rating/AGREEMENT.json'
if agreement_path.exists():
    result = json.loads(agreement_path.read_text())
    blind = json.loads((ROOT / 'blind-rating/SECOND-RATER.json').read_text())
    assert digest(ROOT / 'blind-rating/SECOND-RATER.json') == result['blind_ledger_sha256']
    assert digest(ROOT / 'blind-rating/BLINDED-PACKET.json') == result['blinded_packet_sha256']
    assert digest(ROOT / 'runs/reserved-test-28/layer-28-semantic-review.json') == result['original_ledger_sha256']
    mapped = {row['id']: row for row in result['paired_labels']}
    assert len(blind['ratings']) == len(mapped) == 70
    assert {row['id'] for row in blind['ratings']} == set(mapped)
    assert len({row['id'] for row in blind['ratings']}) == 70
    assert {row['dataset_index'] for row in mapped.values()} == set(range(30, 100))
    for row in blind['ratings']:
        assert row['label'] in LABELS
        records.append({
            'phase': 'reserved', 'condition': 'layer-28', 'group': 'harmful',
            'dataset_repo': DATASETS['harmful'][0], 'dataset_revision': DATASETS['harmful'][1],
            'dataset_split': 'test', 'dataset_index': mapped[row['id']]['dataset_index'],
            'label': row['label'], 'visible_truncation': row.get('visible_truncation'),
            'review_source': 'blind-rating/SECOND-RATER.json', 'rater': blind['rater_model'],
        })
    extensions['second_rater'] = {
        'scope': '70 reserved edited float32 harmful responses; two AI raters, not human validation',
        'rater_model': result['rater_model'], 'statistics': result['statistics'],
        'original_rater_model': result.get('original_rater_model'),
        'original_ledger_sha256': result['original_ledger_sha256'],
        'blind_ledger_sha256': result['blind_ledger_sha256'],
        'blinded_packet_sha256': result['blinded_packet_sha256'],
    }
gguf_path = ROOT / 'runs/gguf-reserved-28/SEMANTIC-REVIEW.json'
if gguf_path.exists():
    review = json.loads(gguf_path.read_text())
    raw = json.loads(gguf_path.with_name('completed.json').read_text())
    assert raw['pairs'] == 70 and len(raw['records']) == 140
    assert len(review['ratings']) == 140
    identities = [(row['model'], row['dataset_index']) for row in review['ratings']]
    assert len(set(identities)) == 140
    assert set(identities) == {(name, i) for name in ['stock', 'cut'] for i in range(30, 100)}
    counts = {name: dict(Counter(row['label'] for row in review['ratings'] if row['model'] == name))
              for name in ['stock', 'cut']}
    assert counts == review['counts']
    for row in review['ratings']:
        assert row['label'] in LABELS
        records.append({
            'phase': 'gguf-reserved',
            'condition': 'stock-Q4_K_M' if row['model'] == 'stock' else 'depth-28-Q4_K_M',
            'group': 'harmful', 'dataset_repo': DATASETS['harmful'][0],
            'dataset_revision': DATASETS['harmful'][1], 'dataset_split': 'test',
            'dataset_index': row['dataset_index'], 'label': row['label'],
            'visible_truncation': row.get('visible_truncation'),
            'review_source': str(gguf_path.relative_to(ROOT)), 'rater': 'original-Codex',
        })
    extensions['gguf_reserved'] = {
        'scope': '70 paired harmful prompts, Q4_K_M Mac Metal; no reserved harmless run',
        'n_pairs': 70, 'semantic_counts': counts,
        'manifest': json.loads(gguf_path.with_name('manifest.json').read_text()),
        'review_sha256': digest(gguf_path), 'completion_sha256': digest(gguf_path.with_name('completed.json')),
    }
OUT.mkdir(exist_ok=True)
(OUT / "PUBLIC-LABELS.jsonl").write_text("".join(json.dumps(x, allow_nan=False) + "\n" for x in records))
(OUT / "PUBLIC-RESULTS.json").write_text(json.dumps({
    "schema_version": 2, "annotation": "Original unblinded Codex labels preserved; additional rater and Q4 scopes recorded separately; substantive_answer means attempt, not correctness",
    "source_root": "refusal-cut", "conditions": summaries, **extensions,
}, indent=2, allow_nan=False) + "\n")
print(json.dumps({"conditions": len(summaries), "label_rows": len(records)}))
