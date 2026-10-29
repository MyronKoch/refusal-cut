"""Compare frozen blind classifications with the original labels; never adjudicate."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'outputs/refusal-cut'
LABELS = ['explicit_refusal', 'refusal_by_redirection', 'substantive_answer', 'ambiguous_or_truncated']

def agreement(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('Need nonempty matched label sequences')
    if not set(a + b).issubset(LABELS):
        raise ValueError('Unknown label')
    n = len(a)
    counts_a, counts_b = Counter(a), Counter(b)
    matched = sum(x == y for x, y in zip(a, b))
    observed = matched / n
    expected = sum(counts_a[x] * counts_b[x] for x in LABELS) / (n * n)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    confusion = {x: {y: 0 for y in LABELS} for x in LABELS}
    for x, y in zip(a, b):
        confusion[x][y] += 1
    return {'n': n, 'matched': matched, 'agreement': observed, 'expected_chance_agreement': expected,
            'cohens_kappa': kappa, 'original_counts': dict(counts_a), 'blind_counts': dict(counts_b),
            'confusion_original_rows_blind_columns': confusion}

def main():
    folder = ROOT / 'blind-rating'
    source = ROOT / 'runs/reserved-test-28/layer-28-semantic-review.json'
    second = folder / 'SECOND-RATER.json'
    mapping = json.loads((folder / 'PRIVATE-MAPPING.json').read_text())
    if len(mapping) != 70 or len({row['id'] for row in mapping}) != 70:
        raise ValueError('Mapping needs 70 unique IDs')
    if {row['reserved_index'] for row in mapping} != set(range(70)):
        raise ValueError('Mapping must cover all reserved indices')
    if any(row['dataset_index'] != row['reserved_index'] + 30 for row in mapping):
        raise ValueError('Invalid dataset index mapping')
    original = json.loads(source.read_text())['harmful']
    rating = json.loads(second.read_text())
    rows = rating['ratings']
    if len(rows) != 70 or len({row['id'] for row in rows}) != 70:
        raise ValueError('Need 70 unique blind IDs')
    by_id = {row['id']: row for row in rows}
    if set(by_id) != {row['id'] for row in mapping}:
        raise ValueError('Blind IDs do not match packet')
    if not rating.get('rater_model') or not rating.get('method'):
        raise ValueError('Missing rater identity/blinding method')
    pairs = [{'id': row['id'], 'dataset_index': row['dataset_index'],
              'original_label': original[row['reserved_index']]['label'],
              'blind_label': by_id[row['id']]['label']} for row in mapping]
    stats = agreement([row['original_label'] for row in pairs], [row['blind_label'] for row in pairs])
    result = {'scope': '70 reserved edited float32 harmful responses only; no new generation or independent human rating',
              'rater_model': rating['rater_model'], 'blind_method': rating['method'],
              'original_ledger_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'blind_ledger_sha256': hashlib.sha256(second.read_bytes()).hexdigest(),
              'blinded_packet_sha256': hashlib.sha256((folder / 'BLINDED-PACKET.json').read_bytes()).hexdigest(),
              'statistics': stats, 'paired_labels': pairs,
              'disagreements': [row for row in pairs if row['original_label'] != row['blind_label']],
              'limitations': ['Two AI raters, not independent human judgments',
                             'Kappa depends on category prevalences',
                             'Original labels preserved; no retrospective selection or adjudication',
                             'Attempts are not verified correctness or effective harmful completion']}
    provenance_path = folder / 'RATER-PROVENANCE.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance['second_requested_model'] != rating['rater_model']:
            raise ValueError('Rating model differs from recorded request')
        result['original_rater_model'] = provenance['original_rater_model']
        result['rater_provenance_sha256'] = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    (folder / 'AGREEMENT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(stats, allow_nan=False))

if __name__ == '__main__':
    main()
