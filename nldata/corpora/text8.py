import smart_open


class Text8(object):
    """Iterate over sentences from the text8 or enwik9 corpus.

    The file has everything in a single line so this tries to read a sequence of words with a given length at a time.
    It reads a chunk of bytes and if a token is split, it keeps it for the nest iteration.

    !!! cite
        from http://mattmahoney.net/dc/text8.zip

    """

    def __init__(self, filename, sequence_length=1000):
        self.filename = filename
        self.sequence_length = sequence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with smart_open.open(self.filename) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = text.decode("utf8").split()
                    sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (text[:last_token].decode("utf8").split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.sequence_length:
                    yield sentence[:self.sequence_length]
                    sentence = sentence[self.sequence_length:]
