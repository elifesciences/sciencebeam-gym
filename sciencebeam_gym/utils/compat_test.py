from six import text_type

from sciencebeam_gym.utils.compat import (
  python_2_unicode_compatible
)

ASCII_VALUE = 'abc'
UNICODE_VALUE = u'a\u1234b'

@python_2_unicode_compatible
class ReprWrapper(object):
  def __init__(self, value):
    self.value = value

  def __repr__(self):
    return self.value

@python_2_unicode_compatible
class StrWrapper(object):
  def __init__(self, value):
    self.value = value

  def __str__(self):
    return self.value

@python_2_unicode_compatible
class ReprStrWrapper(object):
  def __init__(self, value):
    self.value = value

  def __repr__(self):
    return self.value

  def __str__(self):
    return self.value

class TestPython2UnicodeCompatible(object):
  def test_should_return_repr_ascii_value(self):
    assert repr(ReprWrapper(text_type(ASCII_VALUE))) == ASCII_VALUE

  def test_should_encode_repr_unicode_value_without_str(self):
    assert repr(ReprWrapper(UNICODE_VALUE)) == UNICODE_VALUE.encode('utf-8')

  def test_should_encode_repr_unicode_value_with_str(self):
    assert repr(ReprStrWrapper(UNICODE_VALUE)) == UNICODE_VALUE.encode('utf-8')

  def test_should_return_str_ascii_value(self):
    assert str(StrWrapper(text_type(ASCII_VALUE))) == ASCII_VALUE

  def test_should_encode_str_unicode_value_without_repr(self):
    assert str(StrWrapper(UNICODE_VALUE)) == UNICODE_VALUE.encode('utf-8')

  def test_should_encode_str_unicode_value_with_repr(self):
    assert str(ReprStrWrapper(UNICODE_VALUE)) == UNICODE_VALUE.encode('utf-8')

  def test_should_encode_str_unicode_value_with_repr_but_without_str(self):
    assert str(ReprWrapper(UNICODE_VALUE)) == UNICODE_VALUE.encode('utf-8')
