
from __future__ import absolute_import

from apache_beam.io.filesystems import FileSystems

def relative_path(base_path, path):
  if not base_path:
    return path
  if not base_path.endswith('/'):
    base_path += '/'
  return path[len(base_path):] if path.startswith(base_path) else path

def is_relative_path(path):
  return not path.startswith('/') and '://' not in path

def join_if_relative_path(base_path, path):
  return (
    FileSystems.join(base_path, path)
    if base_path and is_relative_path(path)
    else path
  )
