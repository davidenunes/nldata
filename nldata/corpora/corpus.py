from typing import Dict, Iterator, List, Union, Type
from abc import ABC
from functools import partial
from nldata.iterx import only_n_it, chain_it
import os
from itertools import chain


class Corpus(ABC):
    """ Corpus

    !!! note
        you should use `DownloadManager` (from `nldata.utils`) to download and extract files to cache first
        and then use the returning directory to read the dataset if a local data_dir is
        not specified:

        Example:
        ```
            dl_manager = DownloadManager(dataset_name=name)
            data_path = dl_manager.download_and_extract(self.data_url)
            self.data_dir = os.path.join(data_path, "wikitext-103")
        ```


    Args:
        data_dir: directory where the corpus files are located
        data_url: url from which the dataset is to be downloaded

        splits (`Dict[str,str]`): dictionary mapping split names to relative filenames. The splits
        are transformed by merging file names with the provided `data_dir`.

        corpus_it (iterator): an iterator of strings of lists of strings
            that operates on a single corpus file

    """

    def __init__(self,
                 data_dir: str,
                 data_url: str,
                 name: str,
                 splits: Dict[str, str],
                 corpus_it: Type[Iterator[Union[List[str], str]]],
                 **it_args):
        self.data_dir = data_dir
        self.data_url = data_url
        self.name = name
        self.corpus_it = corpus_it
        self.it_args = it_args

        def file_or_files(files):
            if isinstance(files, str):
                return [os.path.join(self.data_dir, files)]
            else:
                return [os.path.join(self.data_dir, file) for file in files]

        self.splits = {name: file_or_files(file) for name, file in splits.items()}

        for name, files in self.splits.items():
            for file in files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"could find file for {name} split in {file}")

    def split(self, name="full", n=None, **kwargs):
        new_args = dict(self.it_args)
        for k in kwargs:
            if k in new_args:
                new_args[k] = kwargs[k]
        iter_split = partial(self.corpus_it, **new_args)

        if name == "full":
            it = chain_it(*(chain_it(*map(iter_split, files)) for files in self.splits.values()))
        else:
            if name not in self.splits:
                raise KeyError(f"invalid split {name}, expected: full, {', '.join(self.splits.keys())}")
            else:
                it = chain_it(*map(iter_split, self.splits[name]))

        if n is not None:
            it = only_n_it(it, n)

        return it
