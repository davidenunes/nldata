import os
import itertools


class PTBIterator:
    EOS_MARK = "<eos>"
    """
    Simple iterator, the file since the file has one sentence per line
    """

    def __init__(self, path, max_samples=None, mark_eos=False):
        assert os.path.exists(path)
        self.path = path
        self.current_sentence = None
        self.source = open(path, 'r')
        self.mark_eos = mark_eos

        self.max_samples = max_samples
        self.num_samples = 0

    def close(self):
        self.file.close()

    def __iter__(self):
        """ Calling iter on the object creates a new iterator with the same settings

        Returns: PTBIterator: an iterator over a given file of the Pen TreeBank corpus

        """
        return PTBIterator(path=self.path, max_samples=self.max_samples, mark_eos=self.mark_eos)

    def __next__(self):
        if self.max_samples is not None and self.num_samples >= self.max_samples:
            self.source.close()
            raise StopIteration

        self.current_sentence = self.source.readline()
        if not self.current_sentence:
            self.source.close()
            raise StopIteration

        self.num_samples += 1
        tokens = self.current_sentence.split()
        if self.mark_eos:
            tokens.append(PTBIterator.EOS_MARK)
        return tokens


class PTB:
    UNKNOWN_TOKEN = "<unk>"

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
            path: path to the directory containing the dataset assets
    """

    def __init__(self, path, mark_eos=False):
        self.mark_eos = mark_eos
        self.train_file = os.path.join(path, 'train.txt')
        self.valid_file = os.path.join(path, 'valid.txt')
        self.test_file = os.path.join(path, 'test.txt')

        if not os.path.exists(self.train_file):
            raise FileNotFoundError("could find train set in {path}".format(path=self.train_file))
        if not os.path.exists(self.valid_file):
            raise FileNotFoundError("could find validation set in {path}".format(path=self.valid_file))
        if not os.path.exists(self.test_file):
            raise FileNotFoundError("could find test set in {path}".format(path=self.test_file))

    def training_set(self, n_samples=None):
        """
        :param n_samples: max number of sentences
        """
        return PTBIterator(path=self.train_file, max_samples=n_samples, mark_eos=self.mark_eos)

    def validation_set(self):
        return PTBIterator(path=self.valid_file, mark_eos=self.mark_eos)

    def test_set(self):
        return PTBIterator(path=self.test_file, mark_eos=self.mark_eos)

    def full(self):
        return itertools.chain(self.training_set(),
                               self.validation_set(),
                               self.test_set())
