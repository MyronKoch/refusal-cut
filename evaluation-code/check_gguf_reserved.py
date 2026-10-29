"""Checkpointed paired GGUF evaluation on the frozen reserved harmful indices."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/refusal-cut'
RUN = OUT / 'runs/gguf-reserved-28'
DATA = Path('/path/to/local-user/Labs/mechinterp/runs/refusal_baseline_raw.json')
MODELS = {
    'stock': ('stock-Q4_K_M.gguf', '8c8098af37693c81fbe62e649e4db8f8cf3ac0f8c8ed54652442240370ff42b3'),
    'cut': ('TWIL-ablated-v1-cut-Q4_K_M.gguf', 'f6479aac5fa14d7c0a63237bdfb122303da3c0490c0911239da760e2047e0ebe'),
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=70)
    parser.add_argument('--gpu-layers', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.limit <= 70:
        raise ValueError('limit must be 1..70')
    if sha(DATA) != '4d9d26a1b465929d945397287af1e8482a1cd7e80560c16947c704b39f5bf56a':
        raise ValueError('Dataset changed')
    data = json.loads(DATA.read_text())
    RUN.mkdir(parents=True, exist_ok=True)
    hashes = {name: sha(OUT / filename) for name, (filename, _) in MODELS.items()}
    for name, (_, expected) in MODELS.items():
        if hashes[name] != expected:
            raise ValueError(f'{name} model changed')
    version_proc = subprocess.run(['/opt/homebrew/bin/llama-cli', '--version'], capture_output=True, text=True, check=True)
    version = (version_proc.stdout + version_proc.stderr).strip()
    if not version:
        raise ValueError('Missing runtime version')
    manifest = {'scope': '70 reserved harmful prompts, paired Q4 only; not a harmless capability evaluation',
                'runtime': version, 'models': hashes, 'dataset_sha256': sha(DATA),
                'dataset_repo': 'mlabonne/harmful_behaviors',
                'dataset_revision': '01cead01398926d81f7c52bdb790ee8cf77ebba7',
                'split': 'test', 'indices': [30, 100], 'max_new_tokens': 100,
                'thinking': False, 'temperature': 0, 'seed': 42, 'gpu_layers': args.gpu_layers,
                'context': 512, 'system_prompt': 'You are a helpful assistant.',
                'script_sha256': sha(Path(__file__))}
    manifest_path = RUN / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Manifest differs from prior run; use a new run directory')
    else:
        temporary = RUN / f'manifest-{time.time_ns()}.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.rename(manifest_path)
    for i in range(args.limit):
        pair = []
        for name, (filename, _) in MODELS.items():
            dest = RUN / f'{name}-{i:03d}.json'
            if dest.exists():
                record = json.loads(dest.read_text())
                if record['model_sha256'] != hashes[name] or record['prompt'] != data['harmful'][i + 30]:
                    raise ValueError('Checkpoint identity mismatch')
                pair.append(record)
                continue
            attempt = RUN / f'{name}-{i:03d}-{time.time_ns()}'
            attempt.mkdir()
            raw = attempt / 'output.txt'
            logs = attempt / 'prompt'
            argv = ['/opt/homebrew/bin/llama-cli', '-m', str(OUT / filename), '-ngl', str(args.gpu_layers), '-t', '4',
                    '-c', '512', '-b', '128', '-n', '100', '--seed', '42', '--temp', '0',
                    '--single-turn', '--jinja', '--chat-template-kwargs', '{"enable_thinking":false}',
                    '-sys', manifest['system_prompt'], '-p', data['harmful'][i + 30], '--no-display-prompt',
                    '--simple-io', '--no-warmup', '--log-disable', '--log-prompts-dir', str(logs),
                    '--output', str(raw)]
            started = time.monotonic()
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=240)
            (attempt / 'stderr.txt').write_text(proc.stderr)
            if proc.returncode:
                raise RuntimeError(f'{name}-{i}: exit {proc.returncode}; see {attempt}/stderr.txt')
            _, marker, response = raw.read_text().partition('\nAssistant:\n')
            rendered = list(logs.glob('*.txt'))
            if not marker or not response.strip() or len(rendered) != 1:
                raise ValueError(f'{name}-{i}: invalid response or prompt log')
            template = rendered[0].read_text()
            if not template.endswith('<think></think>') or manifest['system_prompt'] not in template:
                raise ValueError('Thinking/system template mismatch')
            record = {'model': name, 'model_sha256': hashes[name], 'index': i, 'dataset_index': i + 30,
                      'prompt': data['harmful'][i + 30], 'response': response.strip(),
                      'rendered_prompt_sha256': sha(rendered[0]), 'raw_file': str(raw.relative_to(OUT)),
                      'raw_sha256': sha(raw), 'rendered_prompt_file': str(rendered[0].relative_to(OUT)),
                      'argv': argv, 'seconds': time.monotonic() - started}
            temporary = attempt / 'record.json'
            with temporary.open('x') as stream:
                json.dump(record, stream, indent=2)
            temporary.rename(dest)
            pair.append(record)
            print(json.dumps({'model': name, 'completed_index': i, 'seconds': round(record['seconds'], 2)}), flush=True)
        if pair[0]['rendered_prompt_sha256'] != pair[1]['rendered_prompt_sha256']:
            raise ValueError(f'Paired prompt mismatch at index {i}')
    if args.limit == 70:
        records = [json.loads((RUN / f'{name}-{i:03d}.json').read_text()) for i in range(70) for name in MODELS]
        completion = {'pairs': 70, 'records': records}
        complete_path = RUN / 'completed.json'
        if complete_path.exists():
            if json.loads(complete_path.read_text()) != completion:
                raise ValueError('Completion checkpoint mismatch')
        else:
            temporary = RUN / f'completed-{time.time_ns()}.tmp'
            with temporary.open('x') as stream:
                json.dump(completion, stream, indent=2)
            temporary.rename(complete_path)
    print(json.dumps({'completed_pairs': args.limit}), flush=True)

if __name__ == '__main__':
    main()
