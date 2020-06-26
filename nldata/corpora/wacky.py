# Copyright 2017 Davide Nunes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================================================================
import re
from gzip import open as gzopen
import string

RE_SENTENCE_BEGIN = re.compile("<s>")
RE_SENTENCE_END = re.compile("</s>")
RE_CONTENT_BEGIN = re.compile('<text.*id="(.*)">')
RE_CONTENT_END = re.compile('</text>')
RE_NON_PRINTABLE = re.compile('[^\x96]')
RE_WORD = re.compile('(.*)\t(.*)\t(.*)')


class WaCKy:
    """ WacKy Corpus Reader

        Implements a sentence iterator over WaCKy corpus assets.
        The assets are compressed so we gzip to open them.
        They are also encoded using latin1, so I found it necessary to open them with this encoding (SO-8859-1).

        Args:
            lemma: if True returns the lemmatized version of the corpus

        Returns:
            an Iterator over the sentences in the form of lists of strings.
    """

    def __init__(self, file, lemma=False):
        self.lemma = lemma
        self.current_line = None
        if file.endswith("gz"):
            self.source = gzopen(file, 'rt', encoding='latin-1')
        else:
            self.source = open(file, 'r', encoding='latin-1')

    def close(self):
        self.source.close()

    def __iter__(self):
        return self

    def has_next_sentence(self):
        self.current_line = self.source.readline()
        if not self.current_line:
            return False

        found = False
        while not found:
            self.current_line = self.source.readline()
            if not self.current_line:
                return False

            if RE_SENTENCE_BEGIN.search(self.current_line):
                found = True

        return True

    def __next__(self):
        if not self.has_next_sentence():
            self.source.close()
            raise StopIteration

        done = False
        sentence = []
        while not done:
            self.current_line = self.source.readline()
            if RE_SENTENCE_END.search(self.current_line):
                done = True
            else:
                m = re.search(RE_WORD, self.current_line)
                if m:
                    # (word, tag, lemma)
                    w = m.group(1) if not self.lemma else m.group(3)

                    # correct mixed charset
                    if any(c not in string.printable for c in w):
                        wb = w.encode("latin-1")
                        try:
                            w = wb.decode("windows-1252")
                        except UnicodeDecodeError:
                            # remove weird characters after the word (but considered part of it)
                            if len(w) > 1:
                                w = ''.join(c for c in w if c in string.printable)
                            else:
                                # weird character along, remove it
                                w = " "

                    sentence.append(w)

        return sentence
