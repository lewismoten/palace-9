# Palace-9 raw-history model

## Status

F16, Q6_K (the closest supported substitute for impossible Q8_0), and Q4_K_M each passed their own exhaustive 978,003-case llama.cpp raw-history gate with zero failures: 294,777 legal histories and 683,226 invalid histories.

Copyright 2026 Lewis Moten. Distributed under the Apache License, Version 2.0; see `LICENSE` and `NOTICE`.

A Q8_0 artifact is intentionally absent. Its block-32 requirement is incompatible with this model's 36- and 18-wide narrow tensors; retaining all such tensors as F16 produces no honest Q8_0-quantized tensor set.

## Prompt contract

Submit only a raw move history made from `a` through `i`. The deployment template is literal `<bos>{{ .Prompt }}`. Temperature is `0`; context is `16`; generate exactly one token. A legal history returns an optimal unoccupied square. A malformed, repeated-square, post-win, or out-of-protocol history returns `!`.

## Runtime evidence

`f16-ollama-validation.json` records F16 local Ollama fixtures. Exhaustive raw `/completion` gate reports are retained in `runtime-gates/` for Q6_K and Q4_K_M; the F16 source report and Hugging Face checkpoint are retained in `source-checkpoint/`. Every gate used literal `<bos>` plus raw history, one output token, and temperature zero.

## Scope

This is a 3×3 tic-tac-toe state policy, not a general chat model. It has no external command authority. Do not infer a conventional multi-turn chat protocol from this raw-state completion model.
