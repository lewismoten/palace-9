# PALACE-9

> **Predictive Adversarial Learning and Contingency Evaluator**
>
> A tiny strategic-simulation lab for the least alarming operational theater
> imaginable: a 3×3 tic-tac-toe board.

PALACE-9 is a local, educational mixture-of-experts project. Its hard-won
conclusion under equally capable play is a draw. That is not indecision; it is
what happens when both sides decline to make a strategically unsound move.
No external command authority, weapons, or real-world decision-making is
implemented.

```text
Would you like to play a game?

> abc
> d
```

The live browser lab shows the learned board-state model, its router gates,
expert activations, legality state, and precision comparison. The deterministic
oracle remains the source of optimal-policy labels; the model is evaluated
against it rather than being trusted because it has a dramatic acronym.

## Run the browser lab

```sh
cd /home/ai/tic-tac-toe-moe
python3 -m http.server 18777 --bind 127.0.0.1
# open http://127.0.0.1:18777/
```

The reachable-policy corpus contains **4,519** deduplicated legal positions,
out of a raw `3^9 = 19,683` board-state space.

## The board-state MoE

The canonical browser model is a real trainable board-state mixture of experts,
not a rules engine wearing a neural-network costume.

| Component | Shape / behavior |
|---|---|
| Board-state input | 27 one-hot X/O/empty values plus turn and invalid/duplicate features (`31`) |
| Encoder | `31 → 9` tanh board embedding |
| Shared trunk | `9 → 36 → 36` tanh layers |
| Router | Trainable `36 → 9` softmax gate |
| Experts | Nine generic trainable `36 → 9 → 10` tanh/logit branches |
| Aggregation | Gate-weighted expert logits for nine squares plus `!`, the invalid-history sentinel |

Experts are deliberately unnamed. They are learned conditional-compute paths,
not preassigned “win”, “block”, or “corner” departments. Specialization must be
measured before it earns a tactical label.

### Advanced theater reduction

PALACE-9 does not evaluate every rotation and reflection as a separate crisis.
It applies the eight symmetries of the square (D4): rotations and flips map a
board to a canonical representative for training and evaluation, then map the
chosen move back to the original board coordinates. The corpus reduces from
**4,519 reachable positions to 626 canonical positions**—an **86.15%**
reduction—with no transformed-policy conflicts.

The system has therefore learned the useful strategic lesson that turning the
board upside down does not create a new war.

## Precision is a first-class constraint

The current synchronized browser checkpoint contains **5,958 float-valued
parameters**. It was trained with FP32, FP16, INT8, and per-output-row INT4
forward paths synchronized against the same policy objective. The frozen
browser release reports **0 policy losses, 0 occupied-square selections, and
0 malformed-history sentinel errors** across its strict corpus checks for those
four synchronized paths.

The browser can also simulate lower-bit paths for comparison. These are
**tensor-storage estimates**, not complete files: they exclude JSON, scale,
packing, and runtime overhead. `INT8`/`INT4`/`INT2`/`INT1` labels describe
round/dequantize simulations in the browser, not packed GGUF formats such as
`Q4_K_M`.

| Available precision path | Tensor estimate | Status |
|---|---:|---|
| FP32 / F32 | 23,832 B (23.3 KiB) | Canonical master-weight path |
| FP16 / F16 | 11,916 B (11.6 KiB) | Synchronized and exhaustively validated browser path |
| INT8 / Q8-style | 5,958 B (5.8 KiB) | Synchronized browser path; simulated storage |
| INT4 / Q4-style, per row | 2,979 B (2.9 KiB) | Synchronized browser path; simulated storage |
| INT2 / Q2-style | ~1,490 B (1.5 KiB) | Browser comparison simulation only |
| INT1 / binary-sign | ~745 B | Browser comparison simulation only |

The point is not merely to make the model small after it works. Precision is
part of the experiment from the beginning, so constrained systems can compare
what they save against what they lose.

## Validation contract

A valid next move must:

- remain in the complete equal-value optimal policy set—not just match one
  arbitrary oracle choice;
- never select an occupied square;
- emit `!` for malformed, repeated-square, or post-terminal histories;
- preserve D4 symmetry mapping back to the input board’s original coordinates.

The `!` sentinel is output-only. It is never appended to a game history.

## Deployment track: Hugging Face → GGUF → llama.cpp → Ollama

The browser-native board-state MoE is the canonical evaluator. It is not a
causal Transformer graph and must not be misrepresented as a stock GGUF/Ollama
model.

A separate, deliberately small `Qwen2MoeForCausalLM` deployment track retains
one Transformer layer, 36 hidden units, nine experts, and top-2 routing. The
active tokenizer work replaces the wasteful 151,646-token Qwen vocabulary with
a custom **261-token byte-level GPT-2-compatible vocabulary**: four control
tokens, the 256 byte alphabet, and one inert merge needed by the standard GPT-2
BPE metadata contract. Palace move history is raw text such as `abc`; each move
is one byte token.

A one-epoch Hugging Face export and FP16 GGUF conversion have completed as a
**1,304,384-byte smoke artifact**. It proves the compact export format reaches
the converter. It is not trained enough to publish, load through Ollama, or
claim runtime validation. Full training and exhaustive Hugging Face and
llama.cpp evaluation remain release gates.

When that artifact is verified, create a local Ollama model from the included
`Modelfile`:

```sh
# Edit Modelfile: replace FROM with the absolute verified GGUF path.
ollama create palace-9 -f Modelfile
ollama run palace-9 abc
```

The Modelfile uses `num_ctx 16`: enough for an eight-move prompt and a generated
ninth move. Its raw template intentionally avoids chat framing, separators, or
helpful prose that would change token IDs.

## Key files

| File | Purpose |
|---|---|
| `index.html`, `app.mjs` | Browser lab, inspector, and precision comparison UI |
| `rules.mjs` | Legality checking, minimax oracle, and symmetry transforms |
| `generate_dataset.mjs` | Reachable optimal-policy corpus generation |
| `data/reachable-policy.json` | Source policy corpus |
| `data/default-palace-9-model.json` | Promoted synchronized browser checkpoint |
| `torch_moe_trainer.py` | Browser-model training and frozen validation |
| `torch_qwen2moe_trainer.py` | Separate HF/GGUF deployment-model trainer |
| `Modelfile` | Ollama import template for the future verified GGUF |

## Evaluation over Elo

Elo is useful only for model-versus-model play. PALACE-9’s primary fitness
measure is exhaustive exact-policy and legality validation over the generated
corpus. When perfect play ends in a draw, measuring whether the simulator
avoids unnecessary escalation is more useful than celebrating a winner.
