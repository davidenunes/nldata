import unittest
from nldata.corpora import BNC
import os
from nldata.iterx import chain_it


class TestBNC(unittest.TestCase):
    def test_download_iter(self):
        bnc = BNC()
        it = bnc.split(n=10)
        for sample in it:
            print(sample)


if __name__ == '__main__':
    unittest.main()
