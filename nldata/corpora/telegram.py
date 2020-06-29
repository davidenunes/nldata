from bs4 import BeautifulSoup
import os
from dateutil import parser
from nldata.corpora import Corpus


def get_messages(file):
    soup = BeautifulSoup(file, 'html.parser')
    for msg in soup.find_all('div', {'class': ['message', 'default']}):
        body = msg.find("div", {"class": "body"})
        name = body.find("div", {"class": "from_name"})
        date = body.find("div", {"class": "date"})
        text = body.find("div", {"class": "text"})
        if name and date and text:
            name = str(name.text).strip()
            date = parser.parse(date['title'])
            # date = date.strftime('%d.%m.%Y %H:%M:%S')
            text = str(text.text).strip()
            # print(f"({date.strftime('%H:%M')})  {name}: {text}")
            yield name, date, text


class TelegramIterator:
    EOS = "</s>"
    BOS = "<s>"

    """ Telegram Iterator
    The export dir has multiple html files with N messages per file

    the iterator returns a tuple (name, date, msg)
    """

    def __init__(self, file, max_samples=None):
        assert os.path.exists(file)
        self.file = file
        self.current_message = None
        self.max_samples = max_samples
        self.num_samples = 0
        self.generator = self.generate_samples()

    def generate_samples(self):
        with open(self.file, 'r') as file:
            for name, date, msg in get_messages(file):
                if self.max_samples is not None and self.num_samples >= self.max_samples:
                    return
                else:
                    self.current_message = name.split(" ")[0:1] + [":"] + msg.split(" ")
                    self.current_message = [w for w in self.current_message if w != ""]
                    self.num_samples += 1
                    yield self.current_message

    def __iter__(self):
        return iter(self.generator)

    def __next__(self):
        return next(self.generator)


class Telegram(Corpus):
    def __init__(self, data_dir):
        file_list = list(sorted((f for f in os.listdir(data_dir) if f.endswith(".html"))))
        splits = {"train": file_list}

        super().__init__(data_dir=data_dir,
                         data_url=None,
                         name="Telegram",
                         splits=splits,
                         corpus_it=TelegramIterator)


__all__ = [
    "Telegram"
]
