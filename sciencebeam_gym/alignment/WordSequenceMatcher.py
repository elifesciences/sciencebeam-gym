import logging
from difflib import SequenceMatcher

DEFAULT_SEPARATORS = ' ,'

def get_logger():
  return logging.getLogger(__name__)

def split_with_offset(s, sep):
  previous_start = 0
  tokens = []
  for i, c in enumerate(s):
    if c in sep:
      if previous_start < i:
        tokens.append((previous_start, s[previous_start:i]))
      previous_start = i + 1
  if previous_start < len(s):
    tokens.append((previous_start, s[previous_start:]))
  return tokens

class WordSequenceMatcher(object):
  def __init__(self, isjunk=None, a=None, b=None, sep=None):
    if isjunk:
      raise ValueError('isjunk not supported')
    self.a = a
    self.b = b
    self.sep = sep or DEFAULT_SEPARATORS

  def get_matching_blocks(self):
    a_words_with_offsets = split_with_offset(self.a, self.sep)
    b_words_with_offsets = split_with_offset(self.b, self.sep)
    a_words = [w for _, w in a_words_with_offsets]
    b_words = [w for _, w in b_words_with_offsets]
    a_indices = [i for i, _ in a_words_with_offsets]
    b_indices = [i for i, _ in b_words_with_offsets]
    sm = SequenceMatcher(None, a_words, b_words)
    matching_blocks = [
      (a_indices[ai], b_indices[bi], len(a_words[ai]))
      for ai, bi, size in sm.get_matching_blocks()
      if size
    ]
    return matching_blocks
