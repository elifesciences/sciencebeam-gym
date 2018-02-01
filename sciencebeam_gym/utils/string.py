from future.utils import python_2_unicode_compatible

@python_2_unicode_compatible
class LazyStr(object):
  def __init__(self, fn):
    self.fn = fn

  def __str__(self):
    return self.fn()
