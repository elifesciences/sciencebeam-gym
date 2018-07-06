from .file_path import (
  relative_path,
  join_if_relative_path
)

class TestRelativePath(object):
  def test_should_return_path_if_base_path_is_none(self):
    assert relative_path(None, 'file') == 'file'

  def test_should_return_path_if_path_outside_base_path(self):
    assert relative_path('/parent', '/other/file') == '/other/file'

  def test_should_return_absolute_path_if_base_path_matches(self):
    assert relative_path('/parent', '/parent/file') == 'file'

class TestJoinIfRelativePath(object):
  def test_should_return_path_if_base_path_is_none(self):
    assert join_if_relative_path(None, 'file') == 'file'

  def test_should_return_path_if_not_relative(self):
    assert join_if_relative_path('/parent', '/other/file') == '/other/file'

  def test_should_return_joined_path_if_relative(self):
    assert join_if_relative_path('/parent', 'file') == '/parent/file'
