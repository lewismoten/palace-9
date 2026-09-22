import unittest

from torch_qwen2moe_trainer import PalaceGpt2Tokenizer, expand_history_orderings, prepare_raw_history_pools


class HistoryOrderingTest(unittest.TestCase):
    def test_expands_equivalent_x_move_orders_without_changing_policy(self):
        source = {
            'history': 'cba',
            'policy': {'d': 0.5, 'e': 0.5},
        }
        expanded = {row['history']: row for row in expand_history_orderings([source])}

        self.assertEqual(set(expanded), {'abc', 'cba'})
        self.assertEqual(expanded['abc']['policy'], source['policy'])
        self.assertEqual(expanded['cba']['policy'], source['policy'])

    def test_prepares_tokenized_pools_once_for_epoch_sampling(self):
        valid, invalid = prepare_raw_history_pools(
            [{'history': 'a', 'policy': {'b': 1.0}}],
            [{'history': 'aa', 'policy': {'!': 1.0}}],
            PalaceGpt2Tokenizer(),
        )

        self.assertEqual(valid[0]['input_ids'][0], 1)
        self.assertEqual(valid[0]['history'], 'a')
        self.assertEqual(invalid[0]['history'], 'aa')
        self.assertEqual(invalid[0]['target_ids'], [4])


if __name__ == '__main__':
    unittest.main()
