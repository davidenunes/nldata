import unittest
import os
from nldata.corpora.ptnews import PTNews, PTNewsIterator
from tqdm import tqdm
from nldata.nlp.ptstop import PT
from nldata.nlp.token import is_punct


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path = os.path.join(os.getenv('HOME'), "data/ptnews/")
        self.test_data = os.path.join(os.path.dirname(__file__), "data")

    def test_ptnews(self):
        corpus_reader = PTNews(self.path)

        it = corpus_reader.split("test", n_articles=1)
        for words in tqdm(it):
            words = [word for word in words if word.lower() not in PT and not is_punct(word)]
            last = words
        # reader = PTNewsIterator(path=corpus_reader.train_file, mark_eoa=True)

        # while reader.num_articles < 3:
        #    print(next(reader))


if __name__ == '__main__':
    unittest.main()
