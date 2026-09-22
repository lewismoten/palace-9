# Palace-9 raw-history model

## Status

F16, Q6_K (the closest supported substitute for impossible Q8_0), and Q4_K_M each passed their own exhaustive 978,003-case llama.cpp raw-history gate with zero failures: 294,777 legal histories and 683,226 invalid histories.

Copyright 2026 Lewis Moten. Distributed under the Apache License, Version 2.0; see `LICENSE` and `NOTICE`.

A Q8_0 artifact is intentionally absent. Its block-32 requirement is incompatible with this model's 36- and 18-wide narrow tensors; retaining all such tensors as F16 produces no honest Q8_0-quantized tensor set.

## Prompt contract

Submit only a raw move history made from `a` through `i`. The deployment template is literal `<bos>{{ .Prompt }}`. Temperature is `0`; context is `16`; generate exactly one token. A legal history returns an optimal unoccupied square. A malformed, repeated-square, post-win, or out-of-protocol history returns `!`.

## Runtime evidence

`f16-ollama-validation.json` records F16 local Ollama fixtures. Exhaustive raw `/completion` gate reports are retained in `runtime-gates/` for Q6_K and Q4_K_M; the F16 source report and Hugging Face checkpoint are retained in `source-checkpoint/`. Every gate used literal `<bos>` plus raw history, one output token, and temperature zero.

## Ollama packaging scope

An `ollama create` import consumes this GGUF and the Modelfile configuration; it
does not import or display the repository README, screenshot, website harness,
or browser JSON envelopes. Those companion files remain available in the source
repository/release package for a person to run or inspect separately.

## Browser demonstration

A playable, weight-inspecting browser harness is live at
[lewismoten.github.io/palace-9](https://lewismoten.github.io/palace-9/).

The `browser/` directory contains FP32, F16, Q6_K, and Q4_K_M JSON envelopes for the matching source checkpoint or GGUF release artifact. They preserve the original tensor byte payloads as base64 rather than serializing expanded decimal weights. The included browser decoder applies F32/F16/Q6_K (and Q4_K when present) storage semantics locally before running the causal forward trace.

The Q6_K and Q4_K_M artifact labels describe the released GGUF files. This particular tiny architecture is mixed-storage: incompatible tensors remain F16/F32, and the browser manifests expose the actual type of every stored tensor rather than claiming that every tensor is Q6_K or Q4_K.

## Acknowledgments

The release's code and documentation were developed with assistance from
**GPT-5.6-terra Med**, accessed through
[Hermes](https://hermes-agent.nousresearch.com/) and using **Honcho** for
context and project-memory support. Lewis Moten remains the project designer,
maintainer, and publisher.

## Scope

This is a 3×3 tic-tac-toe state policy, not a general chat model. It has no external command authority. Do not infer a conventional multi-turn chat protocol from this raw-state completion model.
