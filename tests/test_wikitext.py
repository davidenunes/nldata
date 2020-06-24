import unittest
from nldata.corpora.wiki103 import WikiText103


class MyTestCase(unittest.TestCase):
    def test_something(self):
        wiki = WikiText103()
        it = wiki.training_set(10)
        for sample in it:
            pass


if __name__ == '__main__':
    unittest.main()
