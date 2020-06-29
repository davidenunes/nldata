import unittest
from nldata.corpora import Telegram
import os


class TestTelegram(unittest.TestCase):
    def test_export_iter(self):
        pass
        # telegram = Telegram(data_dir)
        # it = telegram.split("train", n=20)
        # samples = [s for s in it]
        # self.assertEqual(len(samples), 20)
        # list(map(print,samples))


if __name__ == '__main__':
    unittest.main()
