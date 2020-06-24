import os
import itertools
from nldata.utils import DownloadManager, DownloadConfig
from nldata.utils import NL_DATASETS_CACHE

UNKNOWN_TOKEN = "<unk>"
EOS = "<eos>"


# TODO this should all be under the same class but the iter separation allows us to
# peek into the current state (current line) of the generator
class WikiTextIterator:
    """
    Simple iterator, the file since the file has one sentence per line
    """

    def __init__(self, file, max_samples=None, mark_eos=False):
        self.file = file
        self.current_line = None
        self.mark_eos = mark_eos
        self.max_samples = max_samples
        self.num_samples = 0
        self.gen = self.gen_samples()

    def gen_samples(self):
        with open(self.file, 'r', encoding='utf8') as file:
            while True:
                if self.max_samples is not None and self.num_samples >= self.max_samples:
                    return

                self.current_line = file.readline()

                if len(self.current_line) == 0:
                    return

                tokens = self.current_line.split()

                if len(tokens) > 0:
                    self.num_samples += 1
                    if self.mark_eos:
                        tokens.append(EOS)
                    yield tokens

    def __iter__(self):
        return self.gen

    def __next__(self):
        next(self.gen)


class WikiText103:
    """ WikiText103 Corpus Reader
            
        Args:
            path: path to the directory containing the dataset assets.
            mark_eos: if true, adds an extra <eos> token to the end of each sentence.
    """

    def __init__(self, path=None, mark_eos=False):
        self.mark_eos = mark_eos
        self.name = "WikiText103"
        self.data_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
        self.data_dir = "wikitext-103"
        dl_cfg = DownloadConfig(cache_dir=NL_DATASETS_CACHE)
        dl_manager = DownloadManager(dataset_name=self.name,
                                     download_config=dl_cfg)

        data_path = dl_manager.download_and_extract(self.data_url)
        data_dir = os.path.join(data_path, self.data_dir)

        self.train_file = os.path.join(data_dir, 'wiki.train.tokens')
        self.valid_file = os.path.join(data_dir, 'wiki.valid.tokens')
        self.test_file = os.path.join(data_dir, 'wiki.test.tokens')

        if not os.path.exists(self.train_file):
            raise FileNotFoundError("could find train set in {path}".format(path=self.train_file))
        if not os.path.exists(self.valid_file):
            raise FileNotFoundError("could find validation set in {path}".format(path=self.valid_file))
        if not os.path.exists(self.test_file):
            raise FileNotFoundError("could find test set in {path}".format(path=self.test_file))

    def training_set(self, n_samples=None):
        return WikiTextIterator(file=self.train_file, max_samples=n_samples, mark_eos=self.mark_eos)

    def validation_set(self, n_samples=None):
        return WikiTextIterator(file=self.valid_file, max_samples=n_samples, mark_eos=self.mark_eos)

    def test_set(self, n_samples=None):
        return WikiTextIterator(file=self.test_file, max_samples=n_samples, mark_eos=self.mark_eos)

    def full(self):
        return itertools.chain(self.training_set(),
                               self.validation_set(),
                               self.test_set())
