# Noughtlear Proliferation

A small educational mixture-of-experts tic-tac-toe simulation. Its projected outcome under equally capable play is a mutually assured draw, not a claim about real-world strategic systems.

## What works now

- `index.html` + `app.mjs`: a browser-only game and sequence inspector.
- `rules.mjs`: deterministic legality checking, minimax teacher, and visible MoE routing labels.
- `generate_dataset.mjs`: emits `data/reachable-policy.json` from reachable positions.
- `test_rules.mjs`: verifies encoding, tactical wins/blocks, and invalid-input echo behavior.

Run locally:

```sh
cd /home/ai/tic-tac-toe-moe
python3 -m http.server 18777 --bind 127.0.0.1
# open http://127.0.0.1:18777/
```

The generated data currently contains 4,519 deduplicated reachable board positions. The theoretical raw board space is `3^9 = 19,683`.

## Trainable board-state MoE

The active browser model is a real, CPU-Worker-trained mixture of experts:

| component | shape / behavior |
|---|---|
| board-state input | 27 one-hot X/O/empty values plus turn and invalid/duplicate features (`31`) |
| trainable encoder | `31 → 9` tanh board embedding |
| shared trunk | `9 → 36 → 36` tanh layers |
| router | trainable `36 → 9` softmax gate |
| experts | nine generic trainable `36 → 9 → 10` tanh/logit branches |
| aggregation | gate-weighted expert logits; nine board squares plus the `!` invalid-history sentinel |

The promoted synchronized checkpoint is evaluated through FP32, FP16, symmetric per-tensor INT8, and symmetric per-output-row INT4 simulations. Each path has 0 policy losses, 0 occupied-square choices, and 0 malformed-history errors on the frozen corpus. The INT formats are dequantize simulations, not GGUF packed-runtime formats.

## Ollama bridge — not yet executed

Ollama needs a supported GGUF architecture and tokenizer; it cannot execute arbitrary browser JavaScript tensors. A correct bridge is:

1. Train the 2-layer policy in-browser against `reachable-policy.json` and export typed-array weights plus provenance.
2. Convert those weights to a compatible, tiny Llama-style Hugging Face checkpoint with an explicit nine-token tokenizer.
3. Use llama.cpp's `convert_hf_to_gguf.py` to create a GGUF.
4. Create the local model with a pinned `Modelfile`:

```text
FROM ./ttt-moe-f16.gguf
PARAMETER num_ctx 8
PARAMETER temperature 0
PARAMETER top_k 1
```

```sh
ollama create ttt-moe-lab -f Modelfile
```

This repository intentionally does **not** perform step 4 yet. The host's converter imports `torch`, but this Python environment currently has neither `torch` nor `safetensors`; and no trained browser checkpoint exists yet. Creating a prompt wrapper from an unrelated model would not be a trained 1–2-layer MoE and would make the Atlas display misleading.

## Evaluation

A correct model is checked against all generated positions for exact next-token match and invalid-history echo behavior. Elo can be added only for policy-vs-policy matchups; exact-policy accuracy is the primary fitness measure because optimal tic-tac-toe play commonly draws.
