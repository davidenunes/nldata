import unittest
import os
from nldata.corpora.ptnews import PTNews, PTNewsIterator
from tqdm import tqdm
from nldata.nlp.ptstop import PT
from nldata.nlp.token import is_punct


# TODO bug EOA is appearing as empty list
class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path = os.path.join(os.getenv('HOME'), "data/ptnews/")
        self.test_data = os.path.join(os.path.dirname(__file__), "data")

    def test_download_iter(self):
        corpus_reader = PTNews()
        it = corpus_reader.split("valid", n_articles=2, mark_eoa=True)
        for words in tqdm(it):
            pass


if __name__ == '__main__':
    unittest.main()
