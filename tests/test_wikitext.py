import unittest
from nldata.corpora.wiki103 import WikiText103
from nldata.iterx import flatten_it


class MyTestCase(unittest.TestCase):
    def test_download_iter(self):
        wiki = WikiText103()
        it = wiki.split(name="test", n=2)
        for sample in it:
            print(sample)


if __name__ == '__main__':
    unittest.main()
