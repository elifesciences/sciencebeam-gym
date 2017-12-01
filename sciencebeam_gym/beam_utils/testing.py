from __future__ import absolute_import

import logging
from contextlib import contextmanager
from io import BytesIO
from mock import patch, Mock, MagicMock
from mock.mock import MagicProxy

import pytest

import apache_beam as beam
from apache_beam.coders.coders import ToStringCoder, StrUtf8Coder
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.io.localfilesystem import LocalFileSystem
from apache_beam.io.filesystem import FileMetadata, MatchResult, CompressionTypes
from apache_beam.io.filesystems import FileSystems


TestPipeline.__test__ = False

_local = {}

def get_logger():
  return logging.getLogger(__name__)

class TestContext(object):
  def __init__(self):
    self.file_content_map = dict()
    self.object_map = dict()

  def set_file_content(self, name, content):
    get_logger().debug('set_file_content: %s (size: %d)', name, len(content))
    self.file_content_map[name] = content

  def get_file_content(self, name):
    return self.file_content_map.get(name)

def get_current_test_context():
  return _local['test_context']

# Apache Beam serialises everything, pretend Mocks being serialised
def unpickle_mock(state):
  get_logger().debug('unpickle mock: state=%s', state)
  obj_id = state[0] if isinstance(state, tuple) else state
  obj = get_current_test_context().object_map[obj_id]
  return obj

unpickle_mock.__safe_for_unpickling__ = True

def mock_reduce(obj):
  obj_id = id(obj)
  get_logger().debug('pickle mock, obj_id: %s', obj_id)
  get_current_test_context().object_map[obj_id] = obj
  return unpickle_mock, (obj_id,)

for c in [Mock, MagicMock, MagicProxy]:
  c.__reduce__ = mock_reduce

@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
class BeamTest(object):
  @pytest.fixture(name='test_context', autouse=True)
  def init_test_context(self):
    get_logger().debug('setting up test context')
    test_context = TestContext()
    _local['test_context'] = test_context
    yield test_context
    get_logger().debug('clearing test context')
    del _local['test_context']

class MockWriteToText(beam.PTransform):
  class WriteDoFn(beam.DoFn):
    def __init__(self, file_path_prefix,
      file_name_suffix='',
      coder=ToStringCoder(),
      header=None):

      self.filename = file_path_prefix + file_name_suffix
      self.file_obj = None
      self.coder = coder
      self.header = header

    def start_bundle(self):
      assert self.filename
      self.file_obj = BytesIO()
      if self.header:
        self.file_obj.write(self.coder.encode(self.header) + '\n')

    def process(self, element):
      assert self.file_obj
      self.file_obj.write(self.coder.encode(element) + '\n')

    def finish_bundle(self):
      assert self.file_obj
      self.file_obj.flush()
      file_content = self.file_obj.getvalue()
      get_logger().debug('file content: %s: %s', self.filename, file_content)
      test_context = get_current_test_context()
      test_context.set_file_content(self.filename, file_content)
      self.file_obj.close()

  def __init__(self, *args, **kwargs):
    self._sink = MockWriteToText.WriteDoFn(*args, **kwargs)

  def expand(self, pcoll):
    return pcoll | 'MockWriteToText' >> beam.ParDo(self._sink)

def MockReadFromText(
  file_pattern=None,
  coder=StrUtf8Coder(),
  skip_header_lines=0):

  file_content = get_current_test_context().get_file_content(file_pattern)
  if file_content is None:
    raise RuntimeError('no file content set for %s' % file_pattern)
  lines = file_content.replace('\r\n', '\n').split('\n')
  if skip_header_lines:
    lines = lines[skip_header_lines:]
  return 'MockReadFromText' >> beam.Create(
    [
      coder.decode(line)
      for line in lines
    ]
  )

class MockFileBasedSource(beam.io.filebasedsource.FileBasedSource):
  def open_file(self, file_name):
    file_content = get_current_test_context().get_file_content(file_name)
    if file_content is None:
      raise RuntimeError('no file content set for %s' % file_name)
    return BytesIO(file_content)

class MockFileSystem(LocalFileSystem):
  @classmethod
  def scheme(cls):
    return 'mock'

  def match(self, patterns, limits=None):
    test_context = get_current_test_context()
    file_content_map = test_context.file_content_map
    all_files = file_content_map.keys()
    if limits is None:
      limits = [None] * len(patterns)
    results = []
    for pattern, limit in zip(patterns, limits):
      files = all_files[:limit]
      metadata = [
        FileMetadata(f, len(file_content_map[f]))
        for f in files
      ]
      results.append(MatchResult(pattern, metadata))
    return results

  def open(
    self, path, mime_type='application/octet-stream',
    compression_type=CompressionTypes.AUTO):

    file_content = get_current_test_context().get_file_content(path)
    if file_content is None:
      raise RuntimeError('no file content set for %s' % path)
    return BytesIO(file_content)

  def create(
    self, path, mime_type='application/octet-stream',
    compression_type=CompressionTypes.AUTO):

    out = BytesIO()
    out.close = lambda: (
      get_current_test_context()
      .set_file_content(path, out.getvalue())
    )
    return out

  def rename(self, source_file_names, destination_file_names):
    test_context = get_current_test_context()
    file_content_map = test_context.file_content_map
    for source_file_name, destination_file_name in zip(source_file_names, destination_file_names):
      get_logger().debug('renaming %s to %s', source_file_name, destination_file_name)
      if source_file_name not in file_content_map:
        raise IOError('mock file does not exist: %s' % source_file_name)
      if destination_file_name in file_content_map:
        raise IOError('mock file already exists: %s' % destination_file_name)
      file_content_map[destination_file_name] = file_content_map[source_file_name]
      del file_content_map[source_file_name]

  def mkdirs(self, path):
    get_logger().debug('mkdirs: %s (no-op)', path)

def mock_get_filesystem(*_):
  return MockFileSystem()

@contextmanager
def patch_beam_io():
  with patch.object(FileSystems, 'get_filesystem', classmethod(mock_get_filesystem)):
    yield
