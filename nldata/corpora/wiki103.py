import os
import itertools
from nldata.utils import DownloadManager, DownloadConfig
from nldata.corpora.corpus import Corpus
from nldata.utils import NL_DATASETS_CACHE

UNKNOWN_TOKEN = "<unk>"
EOS = "<eos>"


class WikiTextIterator:
    """ WikiText Iterator

    Simple iterator, the file since the file has one sentence per line

    Attributes:
        generator

    !!! note
        having a separate iterator class allows us to peek into the current state
        of the dataset generator/iterator. Using a generator instead of an iterator
        class, makes it easier to manage file open and close.
    """

    def __init__(self, file, n=None, mark_eos=False):
        self.file = file
        self.current_line = None
        self.mark_eos = mark_eos
        self.n = n
        self.num_samples = 0
        self.generator = self.sample_generator()

    def sample_generator(self):
        with open(self.file, 'r', encoding='utf8') as file:
            while True:
                if self.n is not None and self.num_samples >= self.n:
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
        return self.generator

    def __next__(self):
        next(self.generator)


class WikiText103(Corpus):
    """ WikiText103 Corpus Reader
            
        Args:
            data_dir: path to the directory containing the dataset files.
                If None downloads and extracts the dataset from data_url
                to the nldata cache.
            mark_eos: if true, adds an extra <eos> token to the end of each sentence.
    """

    def __init__(self, data_dir=None, mark_eos=False):
        name = "WikiText103"
        data_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
        # if no local path found, download the dataset
        if data_dir is None:
            dl_manager = DownloadManager(dataset_name=name)
            data_path = dl_manager.download_and_extract(data_url)
            data_dir = os.path.join(data_path, "wikitext-103")

        splits = {"train": 'wiki.train.tokens',
                  "valid": 'wiki.valid.tokens',
                  "test": 'wiki.test.tokens'}

        super().__init__(
            data_dir=data_dir,
            data_url=data_url,
            name=name,
            splits=splits,
            corpus_it=WikiTextIterator,
            mark_eos=mark_eos
        )


__all__ = [
    "WikiText103"
]
