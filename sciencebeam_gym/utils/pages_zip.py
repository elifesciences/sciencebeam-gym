import logging
from zipfile import ZipFile, ZIP_DEFLATED

from apache_beam.io.filesystems import FileSystems

from sciencebeam_gym.beam_utils.io import (
  dirname,
  mkdirs_if_not_exists
)

def get_logger():
  return logging.getLogger(__name__)

def load_pages(filename, page_range=None):
  with FileSystems.open(filename) as f:
    with ZipFile(f) as zf:
      filenames = zf.namelist()
      if page_range:
        filenames = filenames[
          max(0, page_range[0] - 1):
          page_range[1]
        ]
      for filename in filenames:
        with zf.open(filename) as f:
          yield f

def save_pages(output_filename, ext, bytes_by_page):
  mkdirs_if_not_exists(dirname(output_filename))
  with FileSystems.create(output_filename) as f:
    with ZipFile(f, 'w', compression=ZIP_DEFLATED) as zf:
      for i, data in enumerate(bytes_by_page):
        page_filename = 'page-%s%s' % (1 + i, ext)
        get_logger().debug('page_filename: %s', page_filename)
        zf.writestr(page_filename, data)
    return output_filename
