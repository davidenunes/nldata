import unittest
from nldata.corpora import PTB
import os


class MyTestCase(unittest.TestCase):
    def test_download_iter(self):
        ptb = PTB()
        it = ptb.split("valid", n=2)
        for sample in it:
            pass


if __name__ == '__main__':
    unittest.main()
