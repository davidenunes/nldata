import unittest
from nldata.corpora.wiki103 import WikiText103


class TestWikiText(unittest.TestCase):
    def test_download_iter(self):
        wiki = WikiText103()
        it = wiki.split(name="test", n=2)
        samples = [s for s in it]
        self.assertEqual(len(samples), 2)


if __name__ == '__main__':
    unittest.main()
