import os
from nldata.utils import DownloadManager
from nldata.corpora.corpus import Corpus


class PTNewsIterator:
    """
    Simple iterator, the file since the file has one sentence per line
    Args:
        mark_eoa: mark end of article
    """
    EOA = "<eoa>"  # end of article
    EOS = "<eos>"  # end of sentence
    EOP = "<eop>"  # end of paragraph
    UNKNOWN_TOKEN = "<unk>"  # unknown

    def __init__(self, file,
                 n=None,
                 n_articles=None,
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
        self.n = n
        self.n_articles = n_articles
        self.num_articles = 0
        self.num_samples = 0
        self.with_date_url = with_date_url
        self.gen = self.generate_samples()

    def generate_samples(self):
        with open(self.file, 'r', encoding='utf8') as file:
            while True:
                if self.n is not None and self.num_samples >= self.n:
                    return

                elif self.n_articles is not None and self.num_articles >= self.n_articles:
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
                    else:
                        self.num_samples += 1
                        if self.mark_eos:
                            tokens.append(PTNewsIterator.EOS)
                        yield tokens

    def __iter__(self):
        return iter(self.gen)

    def __next__(self):
        next(self.gen)


class PTNews(Corpus):
    """ WikiText103 Corpus Reader
            
        Args:
            data_dir: path to the directory containing the dataset assets.
            mark_eos: if true, adds an extra <eos> token to the end of each sentence.
    """

    def __init__(self,
                 data_dir=None,
                 mark_eos=False,
                 mark_eoa=False,
                 with_date_url=False,
                 n_articles=None):
        name = "PTNews"
        data_url = "https://zenodo.org/record/3908507/files/ptnews.tar.gz?download=1"

        if data_dir is None:
            dl_manager = DownloadManager(dataset_name=name)
            data_dir = dl_manager.download_and_extract(data_url)

        splits = {"train": 'ptnews.train.tokens',
                  "valid": 'ptnews.valid.tokens',
                  "test": 'ptnews.test.tokens'}

        super().__init__(
            data_dir=data_dir,
            data_url=data_url,
            name=name,
            splits=splits,
            corpus_it=PTNewsIterator,
            mark_eos=mark_eos,
            with_date_url=with_date_url,
            n_articles=n_articles,
            mark_eoa=mark_eoa
        )


__all__ = [
    "PTNews"
]
