#!/usr/bin/env python3
"""Measure strategy-feature enrichment in Palace-9 Qwen2-MoE router choices.

This is descriptive routing analysis, not proof that an expert implements a
human-named strategy. It evaluates all legal raw chronological histories.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from torch_qwen2moe_trainer import PalaceGpt2Tokenizer, expand_history_orderings

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'release/palace-9-local-v1/source-checkpoint'
OUT = ROOT / 'analysis/qwen-expert-routing.json'
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def board_for(history: str) -> list[str]:
    board = [''] * 9
    for turn, move in enumerate(history):
        board[ord(move) - ord('a')] = 'X' if turn % 2 == 0 else 'O'
    return board


def wins(board: list[str], mark: str) -> bool:
    return any(all(board[i] == mark for i in line) for line in LINES)


def immediate_moves(board: list[str], mark: str) -> set[int]:
    result = set()
    for index, value in enumerate(board):
        if value:
            continue
        candidate = board.copy()
        candidate[index] = mark
        if wins(candidate, mark):
            result.add(index)
    return result


def fork_moves(board: list[str], mark: str) -> set[int]:
    result = set()
    for index, value in enumerate(board):
        if value:
            continue
        candidate = board.copy()
        candidate[index] = mark
        if len(immediate_moves(candidate, mark)) >= 2:
            result.add(index)
    return result


def features(history: str, policy: dict) -> set[str]:
    board = board_for(history)
    player = 'X' if len(history) % 2 == 0 else 'O'
    opponent = 'O' if player == 'X' else 'X'
    if isinstance(policy, dict) and 'probabilities' in policy:
        policy = {move: probability for move, probability in zip('abcdefghi!', policy['probabilities'])}
    target = {ord(move) - ord('a') for move, probability in policy.items() if probability and move in 'abcdefghi'}
    own_wins = immediate_moves(board, player)
    opponent_wins = immediate_moves(board, opponent)
    own_forks = fork_moves(board, player)
    opponent_forks = fork_moves(board, opponent)
    flags = {f'ply_{len(history)}'}
    if not history:
        flags.add('opening')
    if own_wins:
        flags.add('immediate_win_available')
        if target & own_wins:
            flags.add('policy_takes_win')
    if opponent_wins:
        flags.add('opponent_immediate_threat')
        if target & opponent_wins:
            flags.add('policy_blocks_win')
    if own_forks:
        flags.add('fork_available')
        if target & own_forks:
            flags.add('policy_creates_fork')
    if opponent_forks:
        flags.add('opponent_fork_available')
    if not board[4]:
        flags.add('center_open')
        if 4 in target:
            flags.add('policy_uses_center')
    if target & {0, 2, 6, 8}:
        flags.add('policy_allows_corner')
    if target & {1, 3, 5, 7}:
        flags.add('policy_allows_edge')
    return flags


def main() -> None:
    raw = json.loads((ROOT / 'data/reachable-policy.json').read_text())['examples']
    rows = expand_history_orderings(raw)
    tokenizer = PalaceGpt2Tokenizer()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(device).eval()

    total_top1 = Counter()
    total_top2 = Counter()
    feature_count = Counter()
    feature_top1 = defaultdict(Counter)
    feature_top2 = defaultdict(Counter)
    ply_count = Counter()
    ply_top1 = defaultdict(Counter)
    feature_ply_count = defaultdict(Counter)
    batch_size = 1024
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            sequences = [tokenizer.history_ids(row['history']) for row in batch_rows]
            width = max(map(len, sequences))
            input_ids = torch.tensor([seq + [tokenizer.pad_id] * (width - len(seq)) for seq in sequences], device=device)
            mask = torch.tensor([[1] * len(seq) + [0] * (width - len(seq)) for seq in sequences], device=device)
            output = model(input_ids=input_ids, attention_mask=mask, output_router_logits=True)
            router = output.router_logits[0].view(len(batch_rows), width, -1)
            last = torch.tensor([len(seq) - 1 for seq in sequences], device=device)
            selected = router[torch.arange(len(batch_rows), device=device), last]
            top2 = selected.topk(2, dim=-1).indices.cpu().tolist()
            for row, experts in zip(batch_rows, top2):
                ply = len(row['history'])
                flags = features(row['history'], row['policy'])
                total_top1[experts[0]] += 1
                total_top2.update(experts)
                ply_count[ply] += 1
                ply_top1[ply][experts[0]] += 1
                for flag in flags:
                    feature_count[flag] += 1
                    feature_top1[flag][experts[0]] += 1
                    feature_top2[flag].update(experts)
                    feature_ply_count[flag][ply] += 1
            if (start // batch_size + 1) % 50 == 0:
                print(f'{min(start + batch_size, len(rows))}/{len(rows)}', flush=True)

    n = len(rows)
    baseline = {expert: total_top1[expert] / n for expert in range(9)}
    report_features = {}
    for flag, count in sorted(feature_count.items()):
        top = {str(expert + 1): round(feature_top1[flag][expert] / count, 6) for expert in range(9)}
        expected_same_ply = {expert: sum(
            feature_ply_count[flag][ply] * ply_top1[ply][expert] / ply_count[ply]
            for ply in feature_ply_count[flag]
        ) / count for expert in range(9)}
        rank = sorted(range(9), key=lambda expert: feature_top1[flag][expert] / count - expected_same_ply[expert], reverse=True)
        winner = rank[0]
        observed = feature_top1[flag][winner] / count
        report_features[flag] = {
            'histories': count,
            'top1_share': top,
            'top2_selection_share': {str(expert + 1): round(feature_top2[flag][expert] / count, 6) for expert in range(9)},
            'most_enriched_top1_expert': winner + 1,
            'top1_share_delta_vs_global': round(observed - baseline[winner], 6),
            'top1_enrichment_ratio_vs_global': round(observed / baseline[winner], 4) if baseline[winner] else None,
            'expected_top1_share_same_ply': round(expected_same_ply[winner], 6),
            'top1_share_delta_vs_same_ply': round(observed - expected_same_ply[winner], 6),
            'top1_enrichment_ratio_vs_same_ply': round(observed / expected_same_ply[winner], 4) if expected_same_ply[winner] else None,
        }
    strategy_flags = ['opening', 'immediate_win_available', 'policy_takes_win', 'opponent_immediate_threat', 'policy_blocks_win', 'fork_available', 'policy_creates_fork', 'opponent_fork_available', 'center_open', 'policy_uses_center', 'policy_allows_corner', 'policy_allows_edge']
    result = {
        'scope': 'All legal raw chronological histories; router output at each history final token.',
        'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
        'histories': n,
        'device': str(device),
        'router_rule': 'top-1 and top-2 ranking from the 9 raw router logits; not a causal claim about isolated expert behavior.',
        'global_top1_share': {str(expert + 1): round(total_top1[expert] / n, 6) for expert in range(9)},
        'global_top2_selection_share': {str(expert + 1): round(total_top2[expert] / n, 6) for expert in range(9)},
        'strategy_feature_results': {flag: report_features[flag] for flag in strategy_flags if flag in report_features},
        'all_feature_results': report_features,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'histories': n,
        'device': str(device),
        'global_top1_share': result['global_top1_share'],
        'strategy_enrichment': {flag: {
            'n': report_features[flag]['histories'],
            'expert': report_features[flag]['most_enriched_top1_expert'],
            'ratio_same_ply': report_features[flag]['top1_enrichment_ratio_vs_same_ply'],
            'delta_same_ply': report_features[flag]['top1_share_delta_vs_same_ply'],
        } for flag in strategy_flags if flag in report_features},
        'report': str(OUT.relative_to(ROOT)),
    }, indent=2))


if __name__ == '__main__':
    main()
