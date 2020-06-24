import xml.etree.ElementTree as ET
import os


class BNCIterator:
    """ BNC Corpus Reader

        Implements a sentence iterator over BNC corpus assets.
        This allows for:
            Iterate over the corpus returning sentences in the form of lists of strings
    """

    def __init__(self, filename):
        self.root = ET.parse(filename)
        self.source = open(filename)
        self.sentence_nodes = self.root.iterfind('.//s')

    def __iter__(self):
        return self

    def __next__(self):
        # raises iteration stop if it doesn't have a next
        try:
            current_sentence_node = next(self.sentence_nodes)
            sentence = [token.strip() for token in current_sentence_node.itertext()]

            return sentence
        except StopIteration:
            self.source.close()
            raise StopIteration()


class BNC:
    """ BNC Corpus

    """

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        """ Calling iter on a bnc object returns a new iterator over the data

        Returns:
            an iterator over the BNC corpus
        """
        return BNCIterator(filename=self.filename)


def file_walker(source_dir):
    """Iterates over xml assets given a source directory (/Texts)
    doesn't do it by file name order
    adding sorted works but loads all file names
    """
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if name.endswith(".xml"):
                yield os.path.join(root, name)
