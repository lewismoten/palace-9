import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from torch_qwen2moe_trainer import QwenPalaceTokenizer, build_qwen_training_examples, collate_sparse_examples


class QwenTokenizerForLlamaCppTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = QwenPalaceTokenizer()

    def test_uses_the_stock_qwen2_tokenizer(self):
        self.assertEqual(self.tokenizer.hf.__class__.__name__, 'Qwen2TokenizerFast')
        self.assertEqual(self.tokenizer.vocab_size, len(self.tokenizer.hf))

    def test_history_and_next_move_use_the_same_spaced_text_contract(self):
        example = build_qwen_training_examples([
            {'history': 'ab', 'policy': {'c': 1.0}},
        ], self.tokenizer)[0]
        self.assertEqual(self.tokenizer.hf.decode(example['input_ids'], skip_special_tokens=True), 'a b')
        self.assertEqual(self.tokenizer.hf.decode([example['target_ids'][0]], skip_special_tokens=True), ' c')

    def test_sparse_targets_do_not_allocate_a_vocab_wide_matrix(self):
        examples = build_qwen_training_examples([
            {'history': '', 'policy': {'a': 0.5, 'e': 0.5}},
            {'history': 'a', 'policy': {'b': 1.0}},
        ], self.tokenizer)
        batch = collate_sparse_examples(examples, self.tokenizer.pad_id)
        self.assertEqual(tuple(batch['target_ids'].shape), (2, 2))
        self.assertEqual(tuple(batch['target_probabilities'].shape), (2, 2))
        self.assertEqual(batch['target_mask'].tolist(), [[True, True], [True, False]])
        self.assertLess(batch['target_ids'].numel(), self.tokenizer.vocab_size)


if __name__ == '__main__':
    unittest.main()
