import json
import shutil
import tempfile
import unittest
from pathlib import Path

from torch_qwen2moe_trainer import PalaceTokenizer, build_model, build_training_examples, collate_examples, forward_last_logits, export_huggingface, load_or_build_model


class Qwen2MoeDeploymentTest(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix='palace9-qwen2moe-test-'))

    def tearDown(self):
        shutil.rmtree(self.temp)

    def test_small_standard_qwen2moe_exports_loadable_huggingface_artifact(self):
        tokenizer = PalaceTokenizer()
        model = build_model(tokenizer.vocab_size)
        exported = self.temp / 'model'
        export_huggingface(model, tokenizer, exported)

        config = json.loads((exported / 'config.json').read_text())
        self.assertEqual(config['model_type'], 'qwen2_moe')
        self.assertEqual(config['vocab_size'], tokenizer.vocab_size)
        self.assertEqual(config['num_experts'], 9)
        self.assertTrue((exported / 'model.safetensors').is_file())
        self.assertTrue((exported / 'tokenizer.json').is_file())

        from transformers import AutoModelForCausalLM
        loaded = AutoModelForCausalLM.from_pretrained(exported)
        self.assertEqual(loaded.config.model_type, 'qwen2_moe')

    def test_resume_loader_restores_a_huggingface_checkpoint(self):
        tokenizer = PalaceTokenizer()
        source = self.temp / 'resume-source'
        export_huggingface(build_model(tokenizer.vocab_size), tokenizer, source)

        loaded = load_or_build_model(tokenizer.vocab_size, source)
        self.assertEqual(loaded.config.model_type, 'qwen2_moe')
        self.assertEqual(loaded.config.vocab_size, tokenizer.vocab_size)

    def test_training_examples_keep_complete_optimal_target_set(self):
        tokenizer = PalaceTokenizer()
        examples = build_training_examples([
            {'history': '', 'policy': {'a': 0.5, 'e': 0.5}},
            {'history': 'aa', 'policy': {'!': 1.0}},
        ], tokenizer)
        self.assertEqual(examples[0]['input_ids'], [tokenizer.token_to_id['<bos>']])
        self.assertEqual(examples[0]['target'][tokenizer.token_to_id['a']], 0.5)
        self.assertEqual(examples[0]['target'][tokenizer.token_to_id['e']], 0.5)
        self.assertEqual(examples[1]['target'][tokenizer.token_to_id['!']], 1.0)

    def test_collation_uses_last_non_padding_token_for_each_history(self):
        tokenizer = PalaceTokenizer()
        examples = build_training_examples([
            {'history': 'a', 'policy': {'b': 1.0}},
            {'history': 'abc', 'policy': {'d': 1.0}},
        ], tokenizer)
        batch = collate_examples(examples, tokenizer.token_to_id['<pad>'])
        self.assertEqual(batch['input_ids'].tolist(), [[1, 4, 0, 0], [1, 4, 5, 6]])
        self.assertEqual(batch['last_positions'].tolist(), [1, 3])
        self.assertEqual(tuple(batch['target'].shape), (2, tokenizer.vocab_size))

    def test_forward_reads_each_history_final_position(self):
        tokenizer = PalaceTokenizer()
        examples = build_training_examples([{'history': 'a', 'policy': {'b': 1.0}}], tokenizer)
        logits = forward_last_logits(build_model(tokenizer.vocab_size), collate_examples(examples, 0))
        self.assertEqual(tuple(logits.shape), (1, tokenizer.vocab_size))


if __name__ == '__main__':
    unittest.main()
