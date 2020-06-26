"""
# coding=utf-8
# Copyright 2020 NLData Authors and TensorFlow Datasets Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import json
from dataclasses import dataclass
from typing import Optional, Union, Dict
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile
import tarfile
import gzip
import os
from hashlib import sha256
from filelock import FileLock
import shutil
from . import __version__
import requests
import sys
from tqdm.auto import tqdm
import logging
from contextlib import contextmanager
from functools import partial
import tempfile

logger = logging.getLogger(__name__)

cache_home = os.path.expanduser(
    os.getenv("NL_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "nldata"))
)
default_datasets_cache_path = os.path.join(cache_home, "datasets")
NL_DATASETS_CACHE = Path(os.getenv("NL_DATASETS_CACHE", default_datasets_cache_path))


def get_size_checksum_dict(path: str) -> dict:
    """Compute the file size and the sha256 checksum of a file"""
    m = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            m.update(chunk)
    return {"num_bytes": os.path.getsize(path), "checksum": m.hexdigest()}


def flatten_nested(data_struct):
    """Flatten data struct of obj or `list`/`dict` of obj"""
    if isinstance(data_struct, dict):
        data_struct = list(flatten_nest_dict(data_struct).values())
        if data_struct and isinstance(data_struct[0], (list, tuple)):
            data_struct = [x for sublist in data_struct for x in sublist]
    if isinstance(data_struct, (list, tuple)):
        return data_struct
    # Singleton
    return [data_struct]


def flatten_nest_dict(d):
    """Return the dict with all nested keys flattened joined with '/'."""
    # Use NonMutableDict to ensure there is no collision between features keys
    flat_dict = ImmutableKeyDict()
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update({"{}/{}".format(k, k2): v2 for k2, v2 in flatten_nest_dict(v).items()})
        else:
            flat_dict[k] = v
    return flat_dict


def map_nested(function, data_struct, dict_only=False, map_tuple=False):
    """Apply a function recursively to each element of a nested data struct."""

    # Could add support for more exotic data_struct, like OrderedDict
    if isinstance(data_struct, dict):
        return {k: map_nested(function, v, dict_only, map_tuple) for k, v in data_struct.items()}
    elif not dict_only:
        types = [list]
        if map_tuple:
            types.append(tuple)
        if isinstance(data_struct, tuple(types)):
            mapped = [map_nested(function, v, dict_only, map_tuple) for v in data_struct]
            if isinstance(data_struct, list):
                return mapped
            else:
                return tuple(mapped)
    # Singleton
    return function(data_struct)


class ImmutableKeyDict(dict):
    """ Immutable Key Dictionary

    dict where keys can be added, but not modified.

    Args:
        error_msg: custom error message to be thrown when user tries to
        overwrite an existing key

    Raises:
        error (`ValueError`): if user ties to overwrite a key.

    """

    def __init__(self, *args, **kwargs):
        self._error_msg = kwargs.pop("error_msg", "Trying to overwrite existing key: {key}", )
        if kwargs:
            raise ValueError("ImmutableKeyDict cannot be initialized with kwargs.")
        super(ImmutableKeyDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(self._error_msg.format(key=key))
        return super(ImmutableKeyDict, self).__setitem__(key, value)

    def update(self, other):
        if any(k in self for k in other):
            raise ValueError(self._error_msg.format(key=set(self) & set(other)))
        return super(ImmutableKeyDict, self).update(other)


# automatically add __init__ __repr__ etc
@dataclass
class DownloadConfig:
    """ Config for cached path manager
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incomplete file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and overide the folder where it was extracted.
    """

    cache_dir: Optional[Union[str, Path]] = None
    force_download: bool = False
    resume_download: bool = False
    local_files_only: bool = False
    proxies: Optional[Dict] = None
    user_agent: Optional[str] = None
    extract_compressed_file: bool = False
    force_extract: bool = False


DEFAULT_DL_CONFIG = DownloadConfig(cache_dir=NL_DATASETS_CACHE)


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https", "s3", "gs", "hdfs")


def is_gzip(path: str) -> bool:
    """from https://stackoverflow.com/a/60634210"""
    with gzip.open(path, "r") as fh:
        try:
            fh.read(1)
            return True
        except OSError:
            return False


def hash_url_to_filename(url, etag=None):
    """ hash url to filename

    Converts `url` into a hashed filename in a repeatable way. If [etag](https://en.wikipedia.org/wiki/HTTP_ETag)
    is specified, append its hash to the url hash delimited by a period. If the url ends with .h5 (HDF5) adds '.h5'
    to the name (e.g. Tensorflow uses this extension)
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".py"):
        filename += ".py"

    return filename


def cached_path(url_or_filename, download_config=None, **download_kwargs, ) -> Optional[str]:
    """ cached_path

    Given a URL or a local path; if it's a URL, download the file, cache it, and return the path to the cached file.
    If it's a local path, make sure the file exists and then return the path.

    Returns:
        path (`str`): a local path to a local or downloaded cached file

    Raises:
        `FileNotFoundError`: if file is non-recoverable (non-existent or no cache on disk)
        `ConnectionError`: if url is unreachable and no cache was found on disk
        `ValueError`: if url or filename could not be parsed
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)

    cache_dir = download_config.cache_dir or NL_DATASETS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=download_config.force_download,
            proxies=download_config.proxies,
            resume_download=download_config.resume_download,
            user_agent=download_config.user_agent,
            local_files_only=download_config.local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("Local file {} doesn't exist".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if download_config.extract_compressed_file and output_path is not None:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path) and not is_gzip(output_path):
            return output_path

        # Path where we extract compressed archives
        # We extract in the cache dir, and get the extracted path name by hashing the original path"
        abs_output_path = os.path.abspath(output_path)
        output_path_extracted = os.path.join(cache_dir, hash_url_to_filename(abs_output_path))

        if (
                os.path.isdir(output_path_extracted)
                and os.listdir(output_path_extracted)
                and not download_config.force_extract
        ) or (os.path.isfile(output_path_extracted) and not download_config.force_extract):
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted, exist_ok=True)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            elif is_gzip(output_path):
                os.rmdir(output_path_extracted)
                with gzip.open(output_path, "rb") as gzip_file:
                    with open(output_path_extracted, "wb") as extracted_file:
                        shutil.copyfileobj(gzip_file, extracted_file)
            else:
                raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

        return output_path_extracted

    return output_path


def http_get(url, temp_file, proxies=None, resume_size=0, user_agent=None, cookies=None):
    ua = "datasets/{}; python/{}".format(__version__, sys.version.split()[0])

    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    headers = {"user-agent": ua}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers, cookies=cookies)
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    )
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(
        url,
        cache_dir=None,
        force_download=False,
        proxies=None,
        etag_timeout=10,
        resume_download=False,
        user_agent=None,
        local_files_only=False,
) -> Optional[str]:
    """ get_from_cache

    Given a URL, look for the corresponding file in the local cache. If it's not there, download it then return the
    path to the cached file.

    Returns:
        path (`str`): a local path to a local or downloaded cached file
    Raises:
       Raises:
        `FileNotFoundError`: if file is non-recoverable (non-existent or no cache on disk)
        `ConnectionError`: if url is unreachable and no cache was found on disk
    """
    if cache_dir is None:
        cache_dir = NL_DATASETS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    original_url = url  # Some parameters may be added
    connected = False
    cookies = None
    etag = None
    if not local_files_only:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code == 200:  # ok
                etag = response.headers.get("ETag")
                for k, v in response.cookies.items():
                    # In some edge cases, we need to get a confirmation token
                    if k.startswith("download_warning") and "drive.google.com" in url:
                        url += "&confirm=" + v
                        cookies = response.cookies
                connected = True
            # In some edge cases, head request returns 400 but the connection is actually ok
            elif (response.status_code == 400 and "firebasestorage.googleapis.com" in url) or (
                    response.status_code == 405 and "drive.google.com" in url
            ):
                connected = True
                logger.info("Couldn't get ETag version for url {}".format(url))
        except (EnvironmentError, requests.exceptions.Timeout):
            # not connected
            pass

    filename = hash_url_to_filename(original_url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # connected == False = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    # try to get the last downloaded one
    if not connected:
        if os.path.exists(cache_path):
            return cache_path
        if local_files_only:
            raise FileNotFoundError(
                "Cannot find the requested files in the cached path and outgoing traffic has been"
                " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                " to False."
            )
        raise ConnectionError("Couldn't reach {}".format(url))

    # From now on, connected is True.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            # GET file object
            http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent, cookies=cookies)

        logger.info("storing %s in cache at %s", url, cache_path)
        shutil.move(temp_file.name, cache_path)

        logger.info("creating metadata file for %s", cache_path)
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


class DownloadManager:
    def __init__(self,
                 dataset_name=None,
                 data_dir=None,
                 download_config=DEFAULT_DL_CONFIG,
                 ):
        """Download manager constructor.
        Args:
            data_dir: can be used to specify a manual directory to get the files from.
            cache_dir: `str`, path to directory where downloads are stored.
            extract_dir: `str`, path to directory where artifacts are extracted.
            dataset_name: `str`, name of dataset this instance will be used for. If
                provided, downloads will contain which datasets they were used for.
            force_download: `bool`, default to False. If True, always [re]download.
        """
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._download_config = download_config
        self._recorded_sizes_checksums = {}

    @property
    def manual_dir(self):
        return self._data_dir

    @property
    def downloaded_size(self):
        """ downloaded size

        Returns the total size of downloaded files.
        """
        return sum(checksums_dict["num_bytes"] for checksums_dict in self._recorded_sizes_checksums.values())

    def _record_sizes_checksums(self, url_or_urls, downloaded_path_or_paths):
        """ Record size and checksum of downloaded files.
        """
        flattened_urls_or_urls = flatten_nested(url_or_urls)
        flattened_downloaded_path_or_paths = flatten_nested(downloaded_path_or_paths)
        for url, path in zip(flattened_urls_or_urls, flattened_downloaded_path_or_paths):
            self._recorded_sizes_checksums[url] = get_size_checksum_dict(path)

    def download_custom(self, url_or_urls, custom_download):
        """  Download given urls(s) by calling a given `custom_download` callable.

        Args:
            url_or_urls: url or `list`/`dict` of urls to download and extract. Each
                url is a `str`.
            custom_download: Callable with signature (src_url: str, dst_path: str) -> Any
                as for example `tf.io.gfile.copy`, that lets you download from google storage
        Returns:
            downloaded_path(s): `str`, The downloaded paths matching the given input
                url_or_urls.
        """
        cache_dir = self._download_config.cache_dir or os.path.join(NL_DATASETS_CACHE, "downloads")

        def url_to_downloaded_path(target_url):
            return os.path.join(cache_dir, hash_url_to_filename(target_url))

        downloaded_path_or_paths = map_nested(url_to_downloaded_path, url_or_urls)
        flattened_urls_or_urls = flatten_nested(url_or_urls)
        flattened_downloaded_path_or_paths = flatten_nested(downloaded_path_or_paths)
        for url, path in zip(flattened_urls_or_urls, flattened_downloaded_path_or_paths):
            try:
                get_from_cache(url, cache_dir=cache_dir, local_files_only=True)
                cached = True
            except FileNotFoundError:
                cached = False
            if not cached or self._download_config.force_download:
                custom_download(url, path)
                get_from_cache(url, cache_dir=cache_dir, local_files_only=True)
        self._record_sizes_checksums(url_or_urls, downloaded_path_or_paths)
        return downloaded_path_or_paths

    def download(self, url_or_urls):
        """Download given url(s).
        Args:
            url_or_urls: url or `list`/`dict` of urls to download and extract. Each
                url is a `str`.
        Returns:
            downloaded_path(s): `str`, The downloaded paths matching the given input
                url_or_urls.
        """
        downloaded_path_or_paths = map_nested(
            lambda url: cached_path(url, download_config=self._download_config, ), url_or_urls,
        )
        self._record_sizes_checksums(url_or_urls, downloaded_path_or_paths)
        return downloaded_path_or_paths

    def iter_archive(self, path):
        """Returns iterator over files within archive.
        Args:
            path: path to archive.
        Returns:
            Generator yielding tuple (path_within_archive, file_obj).
            File-Obj are opened in byte mode (io.BufferedReader)
        """
        logger.info("Extracting archive at %s", str(path))
        extracted_path = self.extract(path)
        if os.path.isfile(extracted_path):
            with open(extracted_path, "rb") as file_obj:
                yield extracted_path, file_obj

        # We do this complex absolute/relative scheme to reproduce the API of iter_tar of tfds
        for root, dirs, files in os.walk(extracted_path, topdown=False):
            relative_dir_path = root.replace(os.path.abspath(extracted_path) + "/", "")
            for name in files:
                relative_file_path = os.path.join(relative_dir_path, name)
                absolute_file_path = os.path.join(root, name)
                with open(absolute_file_path, "rb") as file_obj:
                    yield relative_file_path, file_obj

    @staticmethod
    def extract(path_or_paths):
        """Extract given path(s).
        Args:
            path_or_paths: path or `list`/`dict` of path of file to extract. Each
                path is a `str`.
        Returns:
            extracted_path(s): `str`, The extracted paths matching the given input
                path_or_paths.
        """
        return map_nested(
            lambda path: cached_path(path, extract_compressed_file=True, force_extract=False), path_or_paths,
        )

    def download_and_extract(self, url_or_urls):
        """Download and extract given url_or_urls.
        Is roughly equivalent to:
        ```
        extracted_paths = dl_manager.extract(dl_manager.download(url_or_urls))
        ```
        Args:
            url_or_urls: url or `list`/`dict` of urls to download and extract. Each
                url is a `str`.
        Returns:
            extracted_path(s): `str`, extracted paths of given URL(s).
        """
        return self.extract(self.download(url_or_urls))

    def get_recorded_sizes_checksums(self):
        return self._recorded_sizes_checksums.copy()
