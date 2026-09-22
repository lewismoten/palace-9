import unittest
from pathlib import Path


class ModelfileProtocolTest(unittest.TestCase):
    def test_raw_history_template_includes_the_trained_bos_token(self):
        modelfile = Path(__file__).with_name('Modelfile').read_text()
        self.assertIn('TEMPLATE """<bos>{{ .Prompt }}"""', modelfile)


if __name__ == '__main__':
    unittest.main()
