from __future__ import absolute_import

from io import BytesIO
import logging

from apache_beam.io.filesystems import FileSystems

DEFAULT_BUFFER_SIZE = 4096 * 1024

def get_logger():
  return logging.getLogger(__name__)

def read_all_from_path(path, buffer_size=DEFAULT_BUFFER_SIZE):
  with FileSystems.open(path) as f:
    out = BytesIO()
    while True:
      buf = f.read(buffer_size)
      if not buf:
        break
      out.write(buf)
    return out.getvalue()

def dirname(path):
  return FileSystems.split(path)[0]

def basename(path):
  return FileSystems.split(path)[1]

def find_matching_filenames(pattern):
  return (x.path for x in FileSystems.match([pattern])[0].metadata_list)

def mkdirs_if_not_exists(path):
  if not FileSystems.exists(path):
    try:
      get_logger().info('attempting to create directory: %s', path)
      FileSystems.mkdirs(path)
    except IOError:
      if not FileSystems.exists(path):
        raise

def save_file_content(output_filename, data):
  mkdirs_if_not_exists(dirname(output_filename))
  # Note: FileSystems.create transparently handles compression based on the file extension
  with FileSystems.create(output_filename) as f:
    f.write(data)
  return output_filename
