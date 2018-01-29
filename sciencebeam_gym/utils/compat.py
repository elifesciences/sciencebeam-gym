from six import PY3

def python_2_unicode_compatible(cls):
  """
  Same as futures.utils.python_2_unicode_compatible but with support for __repr__
  """
  if not PY3:
    if cls.__repr__ is not object.__repr__:
      unicode_repr = cls.__repr__
      cls.__repr__ = lambda self: unicode_repr(self).encode('utf-8')
    if cls.__str__ is not object.__str__:
      cls.__unicode__ = cls.__str__
      cls.__str__ = lambda self: self.__unicode__().encode('utf-8')
  return cls
