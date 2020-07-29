import unittest
from nldata.corpora import PTB


class MyTestCase(unittest.TestCase):
    def test_download_iter(self):
        ptb = PTB()
        it = ptb.split("train", n=4)
        samples = [s for s in it]
        self.assertEqual(len(samples), 4)

        it = ptb.split("valid", n=4)
        samples = [s for s in it]
        self.assertEqual(len(samples), 4)

        it = ptb.split("test", n=4)
        samples = [s for s in it]
        self.assertEqual(len(samples), 4)

    def test_exhaust_iter(self):
        ptb = PTB()
        it = ptb.split("valid")


if __name__ == '__main__':
    unittest.main()
