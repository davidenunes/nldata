import os
import itertools
import functools


class PTNewsIterator:
    """
    Simple iterator, the file since the file has one sentence per line
    Args:
        mark_eoa: mark end of article
    """
    EOA = "<eoa>"
    EOS = "<eos>"
    EOP = "<eop>"

    def __init__(self, file,
                 max_samples=None,
                 max_articles=None,
                 mark_eos=False,
                 mark_eoa=False,
                 with_date_url=True):
        assert os.path.exists(file)
        self.file = file
        self.current_line = None
        self.reading_header = True
        self.reading_body = False

        self.mark_eos = mark_eos
        self.mark_eoa = mark_eoa
        self.max_samples = max_samples
        self.max_articles = max_articles
        self.num_articles = 0
        self.num_samples = 0
        self.with_date_url = with_date_url
        self.gen = self.generate_samples()

    def generate_samples(self):
        with open(self.file, 'r', encoding='utf8') as file:
            while True:
                if self.max_samples is not None and self.num_samples >= self.max_samples:
                    return

                elif self.max_articles is not None and self.num_articles >= self.max_articles:
                    return

                self.current_line = file.readline()

                if len(self.current_line) == 0:
                    return

                tokens = self.current_line.split()

                if self.reading_header and len(tokens) > 0:
                    title = tokens
                    self.num_samples += 1
                    if self.mark_eos:
                        tokens.append(PTNewsIterator.EOS)

                    if self.with_date_url:
                        # skip 2 lines (url and date)
                        for _ in range(2):
                            file.readline()
                    self.reading_header = not self.reading_header
                    yield title
                else:
                    # skip empty line and switch from reading header to reading corpus
                    if len(tokens) == 0:
                        # skip empty line
                        # start reading body after head or stop reading body at the end of body
                        if self.reading_body:
                            self.reading_header = True
                            self.num_articles += 1
                            self.reading_body = False
                            if self.mark_eoa:
                                yield [PTNewsIterator.EOA]
                        else:
                            self.reading_body = True

                        # return self.__next__()
                    else:
                        self.num_samples += 1
                        if self.mark_eos:
                            tokens.append(PTNewsIterator.EOS)
                        yield tokens

    def __iter__(self):
        return iter(self.gen)

    def __next__(self):
        next(self.gen)


class PTNews:
    """ WikiText103 Corpus Reader
            
        Args:
            path: path to the directory containing the dataset assets.
            mark_eos: if true, adds an extra <eos> token to the end of each sentence.
    """
    # end of article
    EOA = "<eoa>"
    # end of sentence
    EOS = "<eos>"
    # end of paragraph
    EOP = "<eop>"
    UNKNOWN_TOKEN = "<unk>"

    def __init__(self, path, mark_eos=False):
        self.mark_eos = mark_eos
        self.path = path
        self.train_file = os.path.join(path, 'ptnews.train.tokens')
        self.valid_file = os.path.join(path, 'ptnews.valid.tokens')
        self.test_file = os.path.join(path, 'ptnews.test.tokens')

        self.data_url = "https://storage.googleapis.com/nldata_ptnews/ptnews_v1.tar.gz"

        # extract_path = os.path.join(dl_manager.download_and_extract(_URL), "multi-news-original")

        if not os.path.exists(self.train_file):
            raise FileNotFoundError("could find train set in {path}".format(path=self.train_file))

        # if not os.path.exists(self.valid_file):
        #     raise FileNotFoundError("could find validation set in {path}".format(path=self.valid_file))
        # if not os.path.exists(self.test_file):
        #     raise FileNotFoundError("could find test set in {path}".format(path=self.test_file))

    def split(self, name="full", n_samples=None, n_articles=None):
        iter_split = functools.partial(PTNewsIterator,
                                       max_samples=n_samples,
                                       max_articles=n_articles,
                                       mark_eos=self.mark_eos,
                                       with_date_url=False)

        if name == "full":
            return iter_split(self.train_file)  # itertools.chain(map(iter_split, [self.train_file]))
        elif name == "train":
            return iter_split(self.train_file)
        elif name == "validation":
            return iter_split(self.train_file)
        elif name == "test":
            return iter_split(self.train_file)
        else:
            raise KeyError(f"invalid split {name}, expected: full, train, validation, test")
