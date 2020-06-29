import unittest
from nldata.corpora.ptnews import PTNews
from tqdm import tqdm


class TestPTNews(unittest.TestCase):
    def test_download_iter(self):
        corpus_reader = PTNews()
        it = corpus_reader.split("train", n=4)  # n_articles=2, mark_eoa=True)
        sentences = [s for s in tqdm(it)]
        self.assertEqual(len(sentences), 4)


if __name__ == '__main__':
    unittest.main()
