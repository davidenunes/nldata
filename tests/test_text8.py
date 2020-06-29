import unittest
from nldata.corpora import Text8


class MyTestCase(unittest.TestCase):
    def test_download_iter(self):
        text8 = Text8(sequence_length=4)
        it = text8.split(n=2)
        samples = [s for s in it]
        self.assertEqual(len(samples), 2)


if __name__ == '__main__':
    unittest.main()
