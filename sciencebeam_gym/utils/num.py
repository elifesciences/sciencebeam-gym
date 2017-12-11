from six import raise_from

import numpy as np

def assert_close(a, b, atol=1.e-8):
  try:
    assert np.allclose([a], [b], atol=atol)
  except AssertionError as e:
    raise_from(AssertionError('expected %s to be close to %s (atol=%s)' % (a, b, atol)), e)

def assert_all_close(a, b, atol=1.e-8):
  try:
    assert np.allclose(a, b, atol=atol)
  except AssertionError as e:
    raise_from(AssertionError('expected %s to be close to %s (atol=%s)' % (a, b, atol)), e)
