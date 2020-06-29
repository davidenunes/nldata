import xml.etree.ElementTree as ET
import os
from nldata.corpora import Corpus
from nldata.utils import DownloadManager



def xml_iter(source_dir):
    """Iterates over xml assets given a source directory (/Texts)
    doesn't do it by file name order
    adding sorted works but loads all file names
    """
    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        for name in files:
            if name.endswith(".xml"):
                yield os.path.relpath(os.path.join(root, name), source_dir)


class BNCIterator:
    """ BNC Corpus Reader

        Implements a sentence iterator over BNC corpus assets.
        This allows for:
            Iterate over the corpus returning sentences in the form of lists of strings
    """

    def __init__(self, file):
        self.file = file
        self.generator = self.generate_samples()

    def generate_samples(self):
        with open(self.file, 'r') as file:
            root = ET.parse(self.file)
            sentence_nodes = root.iterfind('.//s')

            while True:
                try:
                    current_sentence_node = next(sentence_nodes)
                    sentence = [token.strip() for token in current_sentence_node.itertext()]

                    yield sentence
                except StopIteration:
                    return

    def __iter__(self):
        return iter(self.generator)

    def __next__(self):
        return next(self.generator)


class BNC(Corpus):
    """ BNC Corpus

    """

    def __init__(self, data_dir=None):
        name = "BNC"
        data_url = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2554/2554.zip?sequence=3&isAllowed=y"
        self.data_dir = data_dir

        dl_manager = DownloadManager(dataset_name=name)
        if data_dir is None:
            data_path = dl_manager.download_and_extract(data_url)
            data_dir = os.path.join(data_path, "download", "Texts")

        split = {"train": xml_iter(data_dir)}

        super().__init__(data_dir=data_dir,
                         data_url=data_url,
                         name=name,
                         splits=split,
                         corpus_it=BNCIterator)

    def __iter__(self):
        """ Calling iter on a bnc object returns a new iterator over the data

        Returns:
            an iterator over the BNC corpus
        """
        return BNCIterator(file=self.filename)


__all__ = [
    "BNC"
]
