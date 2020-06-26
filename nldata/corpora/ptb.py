import os
import itertools
from nldata.corpora import Corpus
from nldata.utils import DownloadManager


class PTBIterator:
    EOS_MARK = "<eos>"
    UNKNOWN_TOKEN = "<unk>"

    """
    Simple iterator, the file since the file has one sentence per line
    """

    def __init__(self, file, n=None, mark_eos=False):
        self.file = file
        self.current_sentence = None
        self.source = open(file, 'r')
        self.mark_eos = mark_eos

        self.max_samples = n
        self.num_samples = 0
        self.generator = self.sample_generator()

    def sample_generator(self):
        with open(self.file, 'r') as file:
            while True:
                if self.max_samples is not None and self.num_samples >= self.max_samples:
                    return

                self.current_sentence = file.readline()
                if len(self.current_sentence) == 0:
                    return

                self.num_samples += 1
                tokens = self.current_sentence.split()
                if self.mark_eos:
                    tokens.append(PTBIterator.EOS_MARK)
                yield tokens

    def __iter__(self):
        return iter(self.generator)

    def __next__(self):
        return next(self.generator)


class PTB(Corpus):
    """ PTB Corpus Reader

        Implements a sentence iterator over PTB WSJ corpus assets. Provided by Mikolov
        with the same pre-processing as in the paper:
         "Empirical Evaluation and Combination of Advanced Language Modeling Techniques"

        This allows for:
            Iterate over the corpus returning sentences in the form of lists of strings
            
        Provides access to iterators for each section of the corpus: full, train, valid, test.
        the train, valid, and test sets are the same as in the paper. 
        
        Splits:
            Sections 0-20 were used as training data (930k tokens), sections 21-22 as validation 
            data (74k tokens) and 23-24 as test data (82k tokens).
            
        Vocab:
            Vocabulary is fixed to 10k unique tokens, words outside this vocabulary are set to
            PTBReader.UNKNOWN_TOKEN
            
        Args:
            data_dir: path to the directory containing the dataset assets
    """

    def __init__(self, data_dir=None, mark_eos=False):
        name = "PTB"
        data_url = "https://zenodo.org/record/3910021/files/ptb.tar.gz?download=1"

        if data_dir is None:
            dl_manager = DownloadManager(dataset_name=name)
            data_dir = dl_manager.download_and_extract(data_url)

        splits = {"train": "ptb.train.txt",
                  "valid": "ptb.valid.txt",
                  "test": "ptb.test.txt"}

        super().__init__(data_dir=data_dir,
                         data_url=data_url,
                         name=name,
                         splits=splits,
                         corpus_it=PTBIterator,
                         mark_eos=mark_eos)


__all__ = [
    "PTB"
]
