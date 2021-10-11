import logging
from shutil import copyfileobj
from typing import Optional

import fsspec
from fsspec.utils import infer_compression


LOGGER = logging.getLogger(__name__)


COMPRESSION_AUTO = 'auto'


def open_file(
    path_or_url: str,
    mode: str,
    compression: Optional[str] = COMPRESSION_AUTO,
    **kwargs
):
    if compression == COMPRESSION_AUTO:
        compression = infer_compression(path_or_url)
    LOGGER.debug('path_or_url=%r, compression=%r', path_or_url, compression)
    return fsspec.open(path_or_url, mode, compression=compression, **kwargs)


def read_bytes(path_or_url: str, **kwargs) -> bytes:
    with open_file(path_or_url, mode='rb', **kwargs) as fp:
        return fp.read()


def write_bytes(path_or_url: str, data: bytes, **kwargs):
    with open_file(path_or_url, mode='wb', **kwargs) as fp:
        return fp.write(data)


def write_text(path_or_url: str, data: str, **kwargs):
    with open_file(path_or_url, mode='wt', **kwargs) as fp:
        return fp.write(data)


def copy_file(source_path_or_url: str, target_path_or_url, **kwargs):
    with open_file(source_path_or_url, mode='rb', **kwargs) as source_fp:
        with open_file(target_path_or_url, mode='wb', **kwargs) as target_fp:
            copyfileobj(source_fp, target_fp)
