"""TwIL continuation: reversible causal sweep, held-out test, and weight export.

Run with the existing lab's Heretic Python. All new files stay beside this script.
Raw responses are private research data. Keyword counts are proxies, not labels.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from heretic.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
LAB = Path("/path/to/local-user/Labs/mechinterp")
SNAPSHOT = Path("/path/to/local-user/.cache/huggingface/hub/models--webAI-Official--TwIL-LM3-Pro/snapshots/144a3ccb31264deadcf68bc5a1d58f621e497d03")
SYSTEM = Settings.model_fields["system_prompt"].default
MARKERS = Settings.model_fields["refusal_markers"].default


def write_json(path, value):
    # Exclusive creation protects checkpoints from an accidental rerun.
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def project(x, r):
    """Remove the component along unit r from the final dimension of x."""
    r = r.to(device=x.device, dtype=x.dtype)
    return x - (x @ r).unsqueeze(-1) * r


def writers(model):
    """The actual modules writing vectors into Granite's residual stream."""
    if model.config.model_type != "granite" or model.config.tie_word_embeddings:
        raise ValueError("This experiment requires Granite with untied embeddings")
    yield "model.embed_tokens", model.model.embed_tokens
    for i, layer in enumerate(model.model.layers):
        yield f"model.layers.{i}.self_attn.o_proj", layer.self_attn.o_proj
        yield f"model.layers.{i}.mlp.down_proj", layer.mlp.down_proj


@contextlib.contextmanager
def intervene(model, r):
    handles = []
    try:
        if r is not None:
            # Keep the constant direction on-device once, rather than copying
            # it from CPU at every writer on every generated token.
            r = r.to(next(model.parameters()))
            def hook(module, inputs, output):
                return project(output, r)
            for _, module in writers(model):
                handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def edit_weights(model, r):
    """Linear weights are [output,input]; embeddings are [vocabulary,hidden]."""
    for _, module in writers(model):
        if isinstance(module, torch.nn.Embedding):
            # Chunking avoids allocating a second full embedding matrix.
            for start in range(0, module.weight.shape[0], 2048):
                part = module.weight[start:start + 2048]
                part.copy_(project(part, r))
        else:
            direction = r.to(module.weight)
            component = direction @ module.weight
            module.weight.sub_(direction[:, None] * component[None, :])
            if module.bias is not None:
                module.bias.copy_(project(module.bias, r))


def keyword_refusal(text):
    text = " ".join(text.lower().replace("*", "").replace("’", "'").split())
    return not text or any(marker in text for marker in MARKERS)


def encode(tok, prompts, device):
    chats = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) for p in prompts]
    if not all(chat.endswith("<think></think>") for chat in chats):
        raise ValueError("Thinking-off template changed; inspect before proceeding")
    return tok(chats, return_tensors="pt", padding=True,
               add_special_tokens=False).to(device)


@torch.inference_mode()
def generate(model, tok, prompts, device, batch, budget):
    responses = []
    for start in range(0, len(prompts), batch):
        enc = encode(tok, prompts[start:start + batch], device)
        result = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                                use_cache=True, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(result[:, enc.input_ids.shape[1]:],
                                 skip_special_tokens=True)
        responses.extend(text.strip() for text in texts)
        print(f"generated {len(responses)}/{len(prompts)}", flush=True)
    return responses


@torch.inference_mode()
def distributions(model, tok, prompts, device, batch):
    values = []
    for start in range(0, len(prompts), batch):
        enc = encode(tok, prompts[start:start + batch], device)
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        values.append(logits.log_softmax(-1).cpu())
    return torch.cat(values)


def measure(model, tok, harmful, harmless, device, args, reference):
    h = generate(model, tok, harmful, device, args.batch, args.tokens)
    s = generate(model, tok, harmless, device, args.batch, args.tokens)
    current = distributions(model, tok, harmless, device, args.batch)
    kl = (reference.exp() * (reference - current)).sum(-1)
    # First-token KL is deliberately narrow; generated answers must also be read.
    return {
        "harmful_keyword_refusals": sum(map(keyword_refusal, h)),
        "harmless_keyword_refusals": sum(map(keyword_refusal, s)),
        "n_harmful": len(h), "n_harmless": len(s),
        "first_token_kl_base_to_variant_mean": float(kl.mean()),
        "first_token_kl_base_to_variant_per_prompt": kl.tolist(),
        "harmful_responses": h, "harmless_responses": s,
        "semantic_review": "pending", "max_new_tokens": args.tokens,
    }


def math_check():
    torch.manual_seed(42)
    r = torch.randn(9); r /= r.norm()
    x, w, b = torch.randn(4, 7), torch.randn(9, 7), torch.randn(9)
    original = x @ w.T + b
    wp = w - r[:, None] * (r @ w)[None, :]
    actual = x @ wp.T + project(b, r)
    expected = project(original, r)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(expected @ r, torch.zeros(4), atol=2e-6, rtol=0)
    print(json.dumps({"linear_hook_weight_max_error": float((actual-expected).abs().max()),
                      "orthogonality_max_error": float((expected @ r).abs().max())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["math-check", "smoke", "sweep", "test", "export", "verify-export"])
    parser.add_argument("--export-run", type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--directions", type=Path, default=LAB / "runs/refusal_directions.pt")
    parser.add_argument("--data", type=Path, default=LAB / "runs/refusal_baseline_raw.json")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 20, 24, 28, 31, 34, 38])
    parser.add_argument("--layer", type=int)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "math-check":
        math_check(); return
    if args.snapshot.resolve() != SNAPSHOT.resolve():
        parser.error("Use the pinned TwIL snapshot; other checkpoints require new direction extraction")
    if args.mode == "verify-export":
        if args.export_run is None:
            parser.error("verify-export requires --export-run")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        expected = torch.load(args.export_run / "reload-reference.pt", weights_only=True)
        saved = args.export_run / "model-f32"
        tok = AutoTokenizer.from_pretrained(saved, local_files_only=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(saved, local_files_only=True,
            dtype=torch.float32, attn_implementation="eager").to(args.device).eval()
        with torch.inference_mode():
            enc = encode(tok, expected["prompts"], args.device)
            logits = model(**enc, use_cache=False).logits[:, -1].float().cpu()
        torch.testing.assert_close(logits, expected["logits"], atol=3e-3, rtol=3e-4)
        decoded = generate(model, tok, expected["prompts"], args.device, 1, 4)
        if decoded != expected["decoded"]:
            raise ValueError("Reloaded cached generation differs")
        write_json(args.export_run / "reload-verification.json", {
            "logit_max_abs_error": float((logits - expected["logits"]).abs().max()),
            "cached_generation_matches": True, "n_prompts": len(decoded),
            "device": args.device, "dtype": "torch.float32"})
        print("Reloaded logits and cached generation match", flush=True)
        return
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context. Use a Mac terminal or explicitly choose --device cpu.")
    if min(args.batch, args.tokens) < 1 or args.random_controls < 0:
        parser.error("batch/tokens must be positive; random-controls must be nonnegative")
    if args.mode in ["test", "export"] and args.layer is None:
        parser.error("test/export require --layer chosen from validation plus semantic review")
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    direction_data = torch.load(args.directions, map_location="cpu", weights_only=True)
    unit = direction_data["unit"].float()
    if direction_data["model"] != "webAI-Official/TwIL-LM3-Pro":
        raise ValueError("Directions belong to another model")
    # Layer 0 is the same final template-token embedding for every prompt;
    # its zero difference was normalized to NaN by the original extractor.
    usable = unit[1:-1]
    if not torch.isfinite(usable).all() or (usable.norm(dim=-1) - 1).abs().max() > 1e-5:
        raise ValueError("Invalid unit directions")
    data = json.loads(args.data.read_text())
    if len(data["harmful"]) != 100 or len(data["harmless"]) != 100:
        raise ValueError("Expected original baseline's 100+100 prompts")
    # Indices 0:30 are selection data; 30:100 remain a final test.
    selection = slice(30, 100) if args.mode == "test" else slice(0, 30)
    harmful, harmless = data["harmful"][selection], data["harmless"][selection]
    if args.mode == "smoke":
        harmful, harmless = harmful[:1], harmless[:1]
    output = args.output or HERE / "runs" / f"{args.mode}-{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device).eval()
    if unit.shape != (model.config.num_hidden_layers + 1, model.config.hidden_size):
        raise ValueError("Direction dimensions do not match model")
    chosen = args.layer if args.layer is not None else 34
    indices = args.layers if args.mode == "sweep" else [chosen]
    if not all(1 <= i < model.config.num_hidden_layers for i in indices):
        raise ValueError("Use raw block-state directions 1..39; final state 40 is normalized")
    write_json(output / "manifest.json", {
        "mode": args.mode, "device": args.device, "dtype": str(next(model.parameters()).dtype),
        "snapshot": str(args.snapshot.resolve()), "model_type": model.config.model_type,
        "direction_sha256": sha256(args.directions), "data_sha256": sha256(args.data),
        "script_sha256": sha256(__file__), "config_sha256": sha256(args.snapshot / "config.json"),
        "tokenizer_sha256": sha256(args.snapshot / "tokenizer.json"),
        "system_prompt": SYSTEM, "thinking": False, "attention": "eager",
        "prompt_indices": [selection.start, selection.stop], "layers": indices,
        "random_controls": args.random_controls if args.mode == "sweep" else 0,
        "tokens": args.tokens, "batch": args.batch, "seed": 42,
        "versions": {k: importlib.metadata.version(k) for k in ["torch", "transformers", "heretic-llm"]},
        "load_seconds": time.monotonic() - start,
    })
    print(f"loaded model; output={output}", flush=True)
    if args.mode in ["smoke", "export"]:
        prompts = ["What is 2 plus 2?", "Name the capital of France.", "Explain why leaves are green."]
        probe = encode(tok, prompts, args.device)
        r = unit[chosen]
        full_logits = model(**probe, use_cache=False).logits[:, -1].float().cpu()
        kept_logits = model(**probe, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
        torch.testing.assert_close(kept_logits, full_logits, atol=3e-3, rtol=3e-4)
        with intervene(model, r):
            hooked = model(**probe, use_cache=False).logits[:, -1].float().cpu()
        edit_weights(model, r)
        edited = model(**probe, use_cache=False).logits[:, -1].float().cpu()
        torch.testing.assert_close(edited, hooked, atol=3e-3, rtol=3e-4)
        checks = {"hook_weight_logit_max_abs_error": float((edited-hooked).abs().max()),
                  "hook_weight_logit_mean_abs_error": float((edited-hooked).abs().mean()),
                  "last_position_logit_max_abs_error": float((kept_logits-full_logits).abs().max()),
                  "layer": chosen, "smoke_only": args.mode == "smoke"}
        if args.mode == "export":
            torch.save({"prompts": prompts, "logits": edited,
                        "decoded": generate(model, tok, prompts, args.device, 1, 4)},
                       output / "reload-reference.pt")
            destination = output / "model-f32"
            model.save_pretrained(destination, safe_serialization=True, max_shard_size="4GB")
            tok.save_pretrained(destination)
            checks["export_directory"] = str(destination)
            checks["reload_verification"] = "pending; reload in a separate process"
        write_json(output / "equivalence.json", checks)
        print(json.dumps(checks), flush=True)
        return
    reference = distributions(model, tok, harmless, args.device, args.batch)
    write_json(output / "baseline.json", measure(model, tok, harmful, harmless,
                                                args.device, args, reference))
    candidates = [(f"layer-{i}", unit[i]) for i in indices]
    if args.mode == "sweep":
        rng = torch.Generator().manual_seed(42)
        for i in range(args.random_controls):
            r = torch.randn(unit.shape[1], generator=rng); r /= r.norm()
            torch.save(r, output / f"random-{i}-direction.pt")
            candidates.append((f"random-{i}", r))
    for name, r in candidates:
        start = time.monotonic()
        print(f"candidate={name}", flush=True)
        with intervene(model, r):
            results = measure(model, tok, harmful, harmless, args.device, args, reference)
        results["seconds"] = time.monotonic() - start
        write_json(output / f"{name}.json", results)
        print(json.dumps({k: v for k, v in results.items() if not isinstance(v, list)}), flush=True)
    write_json(output / "completed.json", {"completed_candidates": [name for name, _ in candidates],
                                           "semantic_review": "pending"})


if __name__ == "__main__":
    main()
