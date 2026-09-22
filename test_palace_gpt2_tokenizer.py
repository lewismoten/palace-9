import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from torch_qwen2moe_trainer import PalaceGpt2Tokenizer, build_qwen_training_examples


class PalaceGpt2TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PalaceGpt2Tokenizer()

    def test_uses_a_compact_byte_level_vocabulary(self):
        self.assertEqual(self.tokenizer.vocab_size, 261)
        self.assertEqual(self.tokenizer.pad_id, 0)
        self.assertEqual(self.tokenizer.bos_id, 1)
        self.assertEqual(self.tokenizer.eos_id, 2)

    def test_move_history_round_trips_as_exact_characters(self):
        self.assertEqual(self.tokenizer.decode(self.tokenizer.history_ids('abc')), 'abc')
        self.assertEqual(self.tokenizer.decode([self.tokenizer.move_id('abc', 'd')]), 'd')
        self.assertEqual(self.tokenizer.decode([self.tokenizer.move_id('', '!')]), '!')

    def test_byte_token_ids_are_stable_across_python_hash_seeds(self):
        program = (
            "import json; from torch_qwen2moe_trainer import PalaceGpt2Tokenizer; "
            "t=PalaceGpt2Tokenizer(); print(json.dumps({x:t.move_id('', x) for x in 'abcdefghi!'}))"
        )
        mappings = []
        for seed in ('1', '2'):
            environment = {**os.environ, 'PYTHONHASHSEED': seed}
            output = subprocess.check_output([sys.executable, '-c', program], text=True, env=environment)
            mappings.append(json.loads(output))
        self.assertEqual(mappings[0], mappings[1])

    def test_saved_artifact_loads_with_huggingface(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.tokenizer.hf.save_pretrained(output)
            loaded = AutoTokenizer.from_pretrained(output)
            self.assertEqual(len(loaded), 261)
            self.assertEqual(loaded.decode(loaded.encode('abc', add_special_tokens=False)), 'abc')

    def test_training_targets_stay_sparse(self):
        examples = build_qwen_training_examples([
            {'history': '', 'policy': {'a': 0.5, 'e': 0.5}},
            {'history': 'a', 'policy': {'b': 1.0}},
        ], self.tokenizer)
        self.assertEqual(examples[0]['target_probabilities'], [0.5, 0.5])
        self.assertEqual(len(examples[0]['target_ids']), 2)


if __name__ == '__main__':
    unittest.main()
