from __future__ import absolute_import

from collections import namedtuple
from itertools import groupby

from six import iteritems

flatten = lambda l: [item for sublist in l for item in sublist]

iter_flatten = lambda l: (item for sublist in l for item in sublist)

def filter_truthy(list_of_something):
  return [l for l in list_of_something if l]

def strip_all(list_of_strings):
  return [(s or '').strip() for s in list_of_strings if s]

def remove_key_from_dict(d, key):
  return {k: v for k, v in iteritems(d) if k != key}

def remove_keys_from_dict(d, keys_to_remove):
  if not keys_to_remove:
    return d
  return {
    k: v
    for k, v in iteritems(d)
    if k not in keys_to_remove
  }

def extract_from_dict(d, key, default_value=None):
  return d.get(key, default_value), remove_key_from_dict(d, key)

def extend_dict(d, *other_dicts, **kwargs):
  """
  example:

  extend_dict(d1, d2)

  is equivalent to Python 3 syntax:
  {
    **d1,
    **d2
  }
  """
  d = d.copy()
  for other_dict in other_dicts:
    d.update(other_dict)
  d.update(kwargs)
  return d

def groupby_to_dict(iterable, key):
  return {
    k: list(v)
    for k, v in groupby(iterable, key=key)
  }

def sort_and_groupby_to_dict(iterable, key):
  return groupby_to_dict(sorted(iterable, key=key), key)

def to_namedtuple(*args, **kwargs):
  name = kwargs.pop('name', 'Tuple')
  d = extend_dict(*list(args) + [kwargs])
  return namedtuple(name, d.keys())(**d)
