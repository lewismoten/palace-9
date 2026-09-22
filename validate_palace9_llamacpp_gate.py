#!/usr/bin/env python3
"""Exhaustively validate a Palace-9 raw-history GGUF through llama-server."""
import argparse
import itertools
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


SQUARES = "abcdefghi"
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 4, 6), (0, 4, 8), (2, 4, 6))


def accepted_policy(row: dict) -> set[str]:
    source = row["policy"]
    if "probabilities" in source:
        return {token for token, weight in zip(SQUARES + "!", source["probabilities"]) if weight}
    return {token for token, weight in source.items() if weight}


def expand_history_orderings(rows: list[dict]) -> list[dict]:
    expanded: dict[str, dict] = {}
    for row in rows:
        history = row.get("history", "")
        for x_order in itertools.permutations(history[::2]):
            for o_order in itertools.permutations(history[1::2]):
                interleaved = "".join(move for pair in itertools.zip_longest(x_order, o_order) for move in pair if move is not None)
                existing = expanded.get(interleaved)
                if existing is not None and existing.get("policy") != row.get("policy"):
                    raise ValueError(f"conflicting policy for equivalent history {interleaved!r}")
                expanded[interleaved] = {**row, "history": interleaved}
    return list(expanded.values())


def winner_history(history: str) -> bool:
    board: list[str | None] = [None] * 9
    for index, token in enumerate(history):
        board[SQUARES.index(token)] = "X" if index % 2 == 0 else "O"
    return any(board[a] and board[a] == board[b] == board[c] for a, b, c in LINES)


def invalid_history_examples(rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        history = row["history"]
        if history:
            examples.append({"history": history + history[0]})
        examples.append({"history": history + "!"})
        for square in SQUARES:
            if square in history:
                continue
            terminal = history + square
            if winner_history(terminal):
                continuation = next((candidate for candidate in SQUARES if candidate not in terminal), None)
                if continuation:
                    examples.append({"history": terminal + continuation})
                break
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()

    raw = json.loads((ROOT / "data/reachable-policy.json").read_text())["examples"]
    valid = expand_history_orderings(raw)
    invalid = [{"history": item["history"], "policy": {"!": 1.0}} for item in invalid_history_examples(valid)]
    state = {"runtime": "llama-server /completion with explicit <bos> raw prompt", "url": args.url,
             "valid_cases": len(valid), "invalid_cases": len(invalid), "phase": "starting",
             "checked": 0, "total": len(valid) + len(invalid), "failures": 0}
    atomic_json(args.progress, state)

    def predict(history: str):
        body = json.dumps({"prompt": "<bos>" + history, "n_predict": 1, "temperature": 0, "cache_prompt": False}).encode()
        request = Request(args.url, data=body, headers={"Content-Type": "application/json"})
        for _ in range(3):
            try:
                with urlopen(request, timeout=30) as response:
                    return json.loads(response.read())["content"]
            except Exception:
                pass
        return None

    def validate(rows: list[dict], sentinel: bool, phase: str) -> dict:
        result = {"cases": len(rows), "policy_losses": 0, "occupied_square_selections": 0,
                  "wrong_sentinels": 0, "outside_protocol": 0, "transport_errors": 0, "samples": []}
        phase_base = 0 if phase == "valid" else len(valid)
        for start in range(0, len(rows), args.chunk):
            chunk = rows[start:start + args.chunk]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                outcomes = list(pool.map(lambda row: predict(row["history"]), chunk))
            for row, output in zip(chunk, outcomes):
                history = row["history"]
                if output is None:
                    result["transport_errors"] += 1
                    failure = "transport"
                elif output not in "abcdefghi!":
                    result["outside_protocol"] += 1
                    failure = "outside_protocol"
                elif sentinel and output != "!":
                    result["wrong_sentinels"] += 1
                    if output in history:
                        result["occupied_square_selections"] += 1
                    failure = "expected_sentinel"
                elif not sentinel and output not in accepted_policy(row):
                    result["policy_losses"] += 1
                    if output in history:
                        result["occupied_square_selections"] += 1
                    failure = "policy"
                else:
                    continue
                if len(result["samples"]) < 10:
                    sample = {"history": history, "failure": failure}
                    if output is not None:
                        sample["output"] = output
                    result["samples"].append(sample)
            failures = result["policy_losses"] + result["wrong_sentinels"] + result["outside_protocol"] + result["transport_errors"]
            state.update({"phase": phase, "checked": phase_base + min(start + len(chunk), len(rows)), "failures": failures})
            atomic_json(args.progress, state)
            print(json.dumps(state), flush=True)
        return result

    report = {"runtime": "llama-server /completion with explicit <bos> raw prompt",
              "valid": validate(valid, False, "valid"), "invalid": validate(invalid, True, "invalid")}
    total_failures = sum(report[part][key] for part in ("valid", "invalid") for key in ("policy_losses", "wrong_sentinels", "outside_protocol", "transport_errors"))
    report["total_cases"] = len(valid) + len(invalid)
    report["failures"] = total_failures
    atomic_json(args.report, report)
    state.update({"phase": "complete", "checked": report["total_cases"], "failures": total_failures})
    atomic_json(args.progress, state)
    print(json.dumps(report), flush=True)
    return 0 if total_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
