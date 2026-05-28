#!/usr/bin/env python
"""Add `<robot_action_N>` special tokens to a Qwen VLM checkpoint.

Required as a one-time preprocessing step before benching the FAST action head:
starVLA's QwenFast maps fast-tokenizer ids (0..2047) to `<robot_action_N>` tokens
in the LLM vocab; without these tokens the labels get masked to -100 and the
CE loss becomes None -> fallback to tensor(0.0) -> backward fails.

Why this script exists instead of using starVLA's own
`starVLA/model/modules/vlm/tools/add_qwen_special_tokens/add_special_tokens_to_qwen.py`:

  1. That script hard-codes `Qwen3VLForConditionalGeneration` and silently fails
     on Qwen3.5 (`Qwen3_5ForConditionalGeneration`). This script uses
     `AutoModelForImageTextToText` so it works on either architecture.
  2. That script saves the processor AFTER the augmented tokenizer, which
     causes the processor's bundled (original, un-augmented) tokenizer to
     overwrite our augmented one. This script reverses the order.

Usage:
  python preprocess_qwen_action_tokens.py \
    --source /path/to/Qwen3.5-0.8B \
    --dest playground/Pretrained_models/Qwen3.5-0.8B-Action \
    --tokens-file starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True, help="Path to source Qwen VLM checkpoint dir.")
    p.add_argument("--dest", required=True, help="Output directory for the augmented model.")
    p.add_argument(
        "--tokens-file",
        default="starVLA/model/modules/vlm/tools/add_qwen_special_tokens/fast_tokens.txt",
        help="One token per line. Defaults to starVLA's fast_tokens.txt (2048 <robot_action_N> tokens).",
    )
    p.add_argument("--init-std", type=float, default=0.02,
                   help="Std for normal-init of new embedding rows.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    new_tokens = [line.strip() for line in open(args.tokens_file) if line.strip()]
    print(f"[preproc] loaded {len(new_tokens)} target tokens from {args.tokens_file}")

    tok = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    old_vocab = len(tok)
    n_added = tok.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"[preproc] tokenizer: old_len={old_vocab} new_len={len(tok)} added={n_added}")

    print(f"[preproc] loading model from {args.source} (AutoModelForImageTextToText) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.source, dtype=torch.bfloat16, trust_remote_code=True
    )
    print(f"[preproc] model class = {type(model).__name__}")

    emb = model.get_input_embeddings()
    print(f"[preproc] old input embedding shape = {tuple(emb.weight.shape)}")
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    emb = model.get_input_embeddings()
    print(f"[preproc] new input embedding shape = {tuple(emb.weight.shape)}")

    # Init only the freshly added rows (old_vocab..len(tok)-1) with N(0, init_std).
    # mean_resizing=False above means the rest is whatever resize_token_embeddings did.
    with torch.no_grad():
        nn.init.normal_(emb.weight[old_vocab:], mean=0.0, std=args.init_std)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != emb.weight.data_ptr():
            nn.init.normal_(out_emb.weight[old_vocab:], mean=0.0, std=args.init_std)

    os.makedirs(args.dest, exist_ok=True)
    print(f"[preproc] saving model to {args.dest} ...")
    model.save_pretrained(args.dest, safe_serialization=True)

    # IMPORTANT save order: processor first, augmented tokenizer second.
    # AutoProcessor.save_pretrained() also serializes its internal (original,
    # un-augmented) tokenizer; if we save processor after tok, it clobbers our
    # additions and the augmented vocab silently reverts.
    try:
        proc = AutoProcessor.from_pretrained(args.source, trust_remote_code=True)
        proc.save_pretrained(args.dest)
        print("[preproc] processor saved (carries original tokenizer; about to overwrite it)")
    except Exception as e:
        print(f"[preproc] processor save skipped: {e!r}")
    tok.save_pretrained(args.dest)
    print("[preproc] augmented tokenizer saved (overwriting processor's copy)")

    # Sanity-check by re-loading.
    tok2 = AutoTokenizer.from_pretrained(args.dest, trust_remote_code=True)
    sample_id = tok2.convert_tokens_to_ids(new_tokens[0])
    assert sample_id == old_vocab, (
        f"verify failed: {new_tokens[0]} -> {sample_id}, expected {old_vocab}"
    )
    print(f"[preproc] verify OK: len(tok)={len(tok2)}, {new_tokens[0]} -> id {sample_id}")

    # Mirror starVLA's add_token_id_map.json convention so downstream tools that
    # look for it (e.g. the original add_special_tokens_to_qwen.py output) can read it.
    mapping = {t: tok2.convert_tokens_to_ids(t) for t in new_tokens}
    with open(os.path.join(args.dest, "added_token_id_map.json"), "w") as f:
        json.dump(mapping, f)

    print(f"[preproc] DONE. Use --base_vlm {args.dest} when running the FAST head bench.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
