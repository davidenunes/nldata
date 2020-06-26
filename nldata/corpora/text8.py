import smart_open
from nldata.corpora import Corpus
from nldata.utils import DownloadManager
import os


class Text8Iterator:
    def __init__(self, file, sequence_length=1000):
        self.file = file
        self.sequence_length = sequence_length
        self.current_sentence = None
        self.generator = self.generate_samples()

    def generate_samples(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        self.current_sentence, rest = [], b""
        with smart_open.open(self.file, mode='rb') as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = text.decode("utf8").split()
                    self.current_sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if self.current_sentence:
                        yield self.current_sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (text[:last_token].decode("utf8").split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                self.current_sentence.extend(words)
                while len(self.current_sentence) >= self.sequence_length:
                    yield self.current_sentence[:self.sequence_length]
                    self.current_sentence = self.current_sentence[self.sequence_length:]

    def __iter__(self):
        return iter(self.generator)

    def __next__(self):
        return next(self.generator)


class Text8(Corpus):
    """Iterate over sentences from the text8 or enwik9 corpus.

    The file has everything in a single line so this tries to read a sequence of words with a given length at a time.
    It reads a chunk of bytes and if a token is split, it keeps it for the nest iteration.

    https://cs.fit.edu/~mmahoney/compression/textdata.html

    !!! cite "Dataset From"
        from http://mattmahoney.net/dc/text8.zip

    """

    def __init__(self, data_dir=None, sequence_length=1000):
        name = "text8"
        data_url = "https://cs.fit.edu/~mmahoney/compression/text8.zip"
        self.sequence_length = sequence_length

        if data_dir is None:
            dl_manager = DownloadManager(dataset_name=name)
            data_dir = dl_manager.download_and_extract(data_url)

        splits = {"train": "text8"}

        super().__init__(data_dir=data_dir,
                         data_url=data_url,
                         name=name,
                         splits=splits,
                         corpus_it=Text8Iterator,
                         sequence_length=sequence_length
                         )


__all__ = [
    "Text8"
]
