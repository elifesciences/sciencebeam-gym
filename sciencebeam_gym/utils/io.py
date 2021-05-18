import logging
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
