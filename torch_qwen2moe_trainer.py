"""Train and export a standard Qwen2-MoE Palace-9 deployment model.

This is intentionally a separate causal-token model.  It does not reuse the
custom board-state-MoE tensors, which are not a Transformers/GGUF architecture.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel, WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen2MoeConfig, Qwen2MoeForCausalLM


class PalaceTokenizer:
    """Fixed move-token vocabulary for the causal deployment model."""

    tokens = ('<pad>', '<bos>', '<eos>', '<unk>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', '!')

    def __init__(self):
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

    def as_huggingface(self) -> PreTrainedTokenizerFast:
        backend = Tokenizer(WordLevel(self.token_to_id, unk_token='<unk>'))
        backend.pre_tokenizer = WhitespaceSplit()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token='<bos>',
            eos_token='<eos>',
            unk_token='<unk>',
            pad_token='<pad>',
        )


class PalaceGpt2Tokenizer:
    """A 260-token byte-level GPT-2 tokenizer for stock llama.cpp text paths.

    Four control tokens are followed by the complete byte alphabet. No BPE
    merges are needed: Palace-9 histories are already compact character text.
    """

    tokens = ('<pad>', '<bos>', '<eos>', '<unk>')

    def __init__(self) -> None:
        # ByteLevel.alphabet() is a hash-order-derived collection in this
        # tokenizers release; sort it so exported token IDs are reproducible
        # across processes and training runs.
        byte_tokens = tuple(sorted(ByteLevel.alphabet()))
        vocab = {token: index + len(self.tokens) for index, token in enumerate(byte_tokens)}
        vocab.update({token: index for index, token in enumerate(self.tokens)})
        probe = Tokenizer(BPE(vocab=vocab, merges=[], unk_token='<unk>'))
        probe.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        nul_token = probe.encode('\x00').tokens[0]
        # llama.cpp's GPT-2 runtime requires a merge list. This one is outside
        # the Palace prompt alphabet, so it establishes valid BPE metadata
        # without changing a–i / ! tokenization.
        vocab[nul_token + nul_token] = len(vocab)
        backend = Tokenizer(BPE(vocab=vocab, merges=[(nul_token, nul_token)], unk_token='<unk>'))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = ByteLevelDecoder()
        self.hf = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token='<bos>',
            eos_token='<eos>',
            unk_token='<unk>',
            pad_token='<pad>',
        )
        self.vocab_size = len(self.hf)
        self.pad_id = self.hf.pad_token_id
        self.bos_id = self.hf.bos_token_id
        self.eos_id = self.hf.eos_token_id

    def history_ids(self, history: str) -> list[int]:
        return [self.bos_id] + self.hf.encode(history, add_special_tokens=False)

    def move_id(self, history: str, move: str) -> int:
        ids = self.hf.encode(move, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'move {move!r} does not map to one byte token: {ids}')
        return ids[0]

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(ids, skip_special_tokens=True)


class QwenPalaceTokenizer:
    """Use Qwen2's stock tokenizer so HF and llama.cpp share text semantics.

    A history is textualized as `a b c`; every predicted move is emitted with
    its leading separator (` d`). This keeps the older Qwen smoke/export path.
    """

    source = 'Qwen/Qwen2-0.5B'

    def __init__(self) -> None:
        self.hf = AutoTokenizer.from_pretrained(self.source)
        if self.hf.pad_token_id is None:
            self.hf.pad_token = self.hf.eos_token
        self.vocab_size = len(self.hf)
        self.pad_id = self.hf.pad_token_id
        self.bos_id = self.hf.bos_token_id

    def history_ids(self, history: str) -> list[int]:
        text = ' '.join(history)
        ids = self.hf.encode(text, add_special_tokens=False)
        return ([self.bos_id] if self.bos_id is not None else []) + ids

    def move_id(self, history: str, move: str) -> int:
        prefix = '' if not history else ' '
        ids = self.hf.encode(prefix + move, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'move {move!r} does not map to one Qwen token: {ids}')
        return ids[0]


def expand_history_orderings(rows: list[dict]) -> list[dict]:
    """Expand a board policy over every chronological ordering of each side.

    A board state does not depend on the order in which a player's already
    placed marks were written. A text-facing causal model does, so train every
    equivalent legal history under the same complete policy.
    """
    expanded: dict[str, dict] = {}
    for row in rows:
        history = row.get('history', '')
        x_moves = history[::2]
        o_moves = history[1::2]
        for x_order in itertools.permutations(x_moves):
            for o_order in itertools.permutations(o_moves):
                interleaved = ''.join(
                    move
                    for pair in itertools.zip_longest(x_order, o_order)
                    for move in pair
                    if move is not None
                )
                existing = expanded.get(interleaved)
                if existing is not None and existing.get('policy') != row.get('policy'):
                    raise ValueError(f'conflicting policy for equivalent history {interleaved!r}')
                expanded[interleaved] = {**row, 'history': interleaved}
    return list(expanded.values())


def build_qwen_training_examples(rows: list[dict], tokenizer: QwenPalaceTokenizer) -> list[dict]:
    """Create sparse complete-policy targets in the stock Qwen2 vocabulary."""
    examples = []
    for row in rows:
        history = row.get('history', '')
        policy = row.get('policy', {})
        if isinstance(policy, dict) and 'probabilities' in policy:
            policy = {token: probability for token, probability in zip('abcdefghi!', policy['probabilities']) if probability}
        items = [(tokenizer.move_id(history, move), float(probability)) for move, probability in policy.items() if probability]
        if not items:
            raise ValueError(f'empty policy for {history!r}')
        examples.append({
            'history': history,
            'input_ids': tokenizer.history_ids(history),
            'target_ids': [item[0] for item in items],
            'target_probabilities': [item[1] for item in items],
        })
    return examples


def prepare_raw_history_pools(valid_rows: list[dict], invalid_rows: list[dict], tokenizer: QwenPalaceTokenizer) -> tuple[list[dict], list[dict]]:
    """Encode raw-history pools once, leaving epochs to sample references only."""
    return (
        build_qwen_training_examples(valid_rows, tokenizer),
        build_qwen_training_examples(invalid_rows, tokenizer),
    )


def collate_sparse_examples(examples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad histories and sparse complete optimal-move target sets."""
    width = max(len(row['input_ids']) for row in examples)
    target_width = max(len(row['target_ids']) for row in examples)
    return {
        'input_ids': torch.tensor([row['input_ids'] + [pad_id] * (width - len(row['input_ids'])) for row in examples], dtype=torch.long),
        'attention_mask': torch.tensor([[1] * len(row['input_ids']) + [0] * (width - len(row['input_ids'])) for row in examples], dtype=torch.long),
        'last_positions': torch.tensor([len(row['input_ids']) - 1 for row in examples], dtype=torch.long),
        'target_ids': torch.tensor([row['target_ids'] + [0] * (target_width - len(row['target_ids'])) for row in examples], dtype=torch.long),
        'target_probabilities': torch.tensor([row['target_probabilities'] + [0.0] * (target_width - len(row['target_probabilities'])) for row in examples], dtype=torch.float32),
        'target_mask': torch.tensor([[True] * len(row['target_ids']) + [False] * (target_width - len(row['target_ids'])) for row in examples], dtype=torch.bool),
    }


def build_training_examples(rows: list[dict], tokenizer: PalaceTokenizer) -> list[dict]:
    """Encode histories while retaining every oracle-accepted output token."""
    examples = []
    for row in rows:
        history = row.get('history', '')
        input_ids = [tokenizer.token_to_id['<bos>']] + [
            tokenizer.token_to_id.get(token, tokenizer.token_to_id['<unk>']) for token in history
        ]
        target = [0.0] * tokenizer.vocab_size
        policy = row.get('policy', {})
        if isinstance(policy, dict) and 'probabilities' in policy:
            for index, probability in enumerate(policy['probabilities']):
                if probability:
                    target[tokenizer.token_to_id['abcdefghi!'[index]]] = float(probability)
        else:
            for token, probability in policy.items():
                target[tokenizer.token_to_id[token]] = float(probability)
        examples.append({'history': history, 'input_ids': input_ids, 'target': target})
    return examples


def collate_examples(examples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad variable-length causal histories and retain their final token index."""
    width = max(len(row['input_ids']) for row in examples)
    input_ids = [row['input_ids'] + [pad_id] * (width - len(row['input_ids'])) for row in examples]
    attention_mask = [[1] * len(row['input_ids']) + [0] * (width - len(row['input_ids'])) for row in examples]
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'last_positions': torch.tensor([len(row['input_ids']) - 1 for row in examples], dtype=torch.long),
        'target': torch.tensor([row['target'] for row in examples], dtype=torch.float32),
    }


def build_model(vocab_size: int) -> Qwen2MoeForCausalLM:
    """Construct a deliberately small, stock Transformers Qwen2-MoE model."""
    config = Qwen2MoeConfig(
        vocab_size=vocab_size,
        hidden_size=36,
        intermediate_size=72,
        num_hidden_layers=1,
        num_attention_heads=9,
        num_key_value_heads=3,
        max_position_embeddings=16,
        num_experts=9,
        num_experts_per_tok=2,
        moe_intermediate_size=18,
        decoder_sparse_step=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    return Qwen2MoeForCausalLM(config)


def load_or_build_model(vocab_size: int, resume: Path | None = None) -> Qwen2MoeForCausalLM:
    """Load a compatible HF checkpoint when resuming, otherwise start fresh."""
    if resume is None:
        return build_model(vocab_size)
    model = Qwen2MoeForCausalLM.from_pretrained(resume)
    if model.config.vocab_size != vocab_size:
        raise ValueError(f'resume checkpoint vocabulary {model.config.vocab_size} does not match tokenizer vocabulary {vocab_size}')
    return model


def forward_last_logits(model: Qwen2MoeForCausalLM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the causal model and return logits after each supplied history."""
    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    rows = torch.arange(batch['input_ids'].shape[0], device=output.logits.device)
    return output.logits[rows, batch['last_positions'].to(output.logits.device)]


def export_huggingface(model: Qwen2MoeForCausalLM, tokenizer: PalaceTokenizer, output: Path) -> None:
    """Write a self-contained standard Transformers artifact for conversion."""
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    hf_tokenizer = tokenizer.as_huggingface()
    hf_tokenizer.save_pretrained(output)
    (output / 'palace9-provenance.json').write_text(json.dumps({
        'format': 'palace9-qwen2moe/v1',
        'architecture': 'Qwen2MoeForCausalLM',
        'token_vocabulary': list(tokenizer.tokens),
        'purpose': 'causal next-move token prediction; independently trained deployment model',
    }, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a standard Qwen2-MoE Palace-9 deployment model')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--rate', type=float, default=0.001)
    parser.add_argument('--examples-per-epoch', type=int, default=65536,
                        help='balanced random sample of valid and invalid raw histories per epoch')
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--output', default='palace9-qwen2moe-hf')
    parser.add_argument('--resume', type=Path, help='Hugging Face checkpoint to continue training from')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    raw_rows = json.loads(Path('data/reachable-policy.json').read_text())['examples']
    valid_rows = expand_history_orderings(raw_rows)
    from torch_moe_trainer import invalid_history_examples
    invalid_rows = [{'history': row['history'], 'policy': {'!': 1.0}} for row in invalid_history_examples(valid_rows)]
    if args.examples_per_epoch < 2:
        raise ValueError('--examples-per-epoch must be at least 2')
    valid_per_epoch = args.examples_per_epoch // 2
    invalid_per_epoch = args.examples_per_epoch - valid_per_epoch
    if valid_per_epoch > len(valid_rows) or invalid_per_epoch > len(invalid_rows):
        raise ValueError('--examples-per-epoch exceeds the available raw-history corpus')
    tokenizer = PalaceGpt2Tokenizer()
    valid_examples, invalid_examples = prepare_raw_history_pools(valid_rows, invalid_rows, tokenizer)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / 'training-progress.json'
    model = load_or_build_model(tokenizer.vocab_size, args.resume).to(device)
    model.config.bos_token_id = tokenizer.bos_id
    model.config.eos_token_id = tokenizer.eos_id
    model.config.pad_token_id = tokenizer.pad_id
    model.config.palace9_tokenizer = 'gpt2-byte-v1'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate, weight_decay=0.01)
    for epoch in range(1, args.epochs + 1):
        examples = random.sample(valid_examples, valid_per_epoch) + random.sample(invalid_examples, invalid_per_epoch)
        random.shuffle(examples)
        model.train()
        total_loss = 0.0
        for offset in range(0, len(examples), args.batch_size):
            batch = {name: value.to(device) for name, value in collate_sparse_examples(examples[offset:offset + args.batch_size], tokenizer.pad_id).items()}
            logits = forward_last_logits(model, batch)
            selected_log_probs = torch.log_softmax(logits, dim=-1).gather(1, batch['target_ids'])
            loss = -(selected_log_probs * batch['target_probabilities'] * batch['target_mask']).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch['input_ids'])
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            progress = {'epoch': epoch, 'loss': total_loss / len(examples), 'examples': len(examples), 'device': str(device)}
            status_path.write_text(json.dumps(progress, indent=2) + '\n')
            print(json.dumps(progress), flush=True)
    model.cpu()
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.hf.save_pretrained(destination)
    (destination / 'palace9-provenance.json').write_text(json.dumps({
        'format': 'palace9-qwen2moe/v3',
        'architecture': 'Qwen2MoeForCausalLM',
        'tokenizer': 'palace9-gpt2-byte-v1',
        'vocabulary_size': tokenizer.vocab_size,
        'history_format': 'raw move characters without separators; one byte token per move',
        'purpose': 'causal next-move token prediction; independently trained compact llama.cpp-ready deployment model',
    }, indent=2) + '\n')
    print(json.dumps({'exported': str(destination), 'epochs': args.epochs, 'examples': len(examples), 'tokenizer': 'palace9-gpt2-byte-v1', 'vocabulary_size': tokenizer.vocab_size}), flush=True)


if __name__ == '__main__':
    main()
