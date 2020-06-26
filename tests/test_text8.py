import unittest
from nldata.corpora import Text8
import os


class MyTestCase(unittest.TestCase):
    def test_download_iter(self):
        text8 = Text8(sequence_length=4)
        it = text8.split(n=2)
        for sample in it:
            print(len(sample))
            print(sample)


if __name__ == '__main__':
    unittest.main()
