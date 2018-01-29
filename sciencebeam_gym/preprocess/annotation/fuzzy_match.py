from __future__ import division

import logging

from sciencebeam_gym.utils.string import (
  LazyStr
)

from sciencebeam_gym.alignment.align import (
  LocalSequenceMatcher,
  SimpleScoring
)

from sciencebeam_gym.alignment.WordSequenceMatcher import (
  WordSequenceMatcher
)

DEFAULT_SCORING = SimpleScoring(
  match_score=2,
  mismatch_score=-1,
  gap_score=-2
)

def get_logger():
  return logging.getLogger(__name__)

def len_index_range(index_range):
  return index_range[1] - index_range[0]

# Treat space or comma after a dot, or a dot after a letter as junk
DEFAULT_ISJUNK = lambda s, i: (
  (i > 0 and s[i - 1] == '.' and (s[i] == ' ' or s[i] == ',')) or
  (i > 0 and s[i - 1].isalpha() and s[i] == '.')
)

def invert_index_ranges(range_list, start, end):
  i = start
  for r_start, r_end in range_list:
    if i >= end:
      return
    if i < r_start:
      yield i, min(end, r_start)
    i = r_end
  if i < end:
    yield i, end

class FuzzyMatchResult(object):
  def __init__(self, a, b, matching_blocks, isjunk=None):
    self.a = a
    self.b = b
    self.matching_blocks = matching_blocks
    self.non_empty_matching_blocks = [x for x in self.matching_blocks if x[-1]]
    self._match_count = None
    self._a_index_range = None
    self._b_index_range = None
    self.isjunk = isjunk or DEFAULT_ISJUNK

  def has_match(self):
    return len(self.non_empty_matching_blocks) > 0

  def match_count(self):
    if self._match_count is None:
      self._match_count = sum(triple[-1] for triple in self.matching_blocks)
    return self._match_count

  def ratio_to(self, size):
    if not size:
      return 0.0
    return self.match_count() / size

  def ratio(self):
    a_match_len = len_index_range(self.a_index_range())
    b_match_len = len_index_range(self.b_index_range())
    max_len = max(a_match_len, b_match_len)
    if max_len == a_match_len:
      junk_match_count = self.a_non_matching_junk_count(self.a_index_range())
    else:
      junk_match_count = self.b_non_matching_junk_count(self.b_index_range())
    max_len_excl_junk = max_len - junk_match_count
    result = self.ratio_to(max_len_excl_junk)
    if result > 1.0:
      get_logger().debug(
        'ratio: ratio greater than 1.0, a_match_len=%d, b_match_len=%d, max_len=%d,'
        ' junk_match_count=%d, max_len_excl_junk=%d, result=%f',
        a_match_len, b_match_len, max_len, junk_match_count, max_len_excl_junk, result
      )
    return result

  def count_junk_between(self, s, index_range):
    if not self.isjunk:
      return 0
    return sum(self.isjunk(s, i) for i in range(index_range[0], index_range[1]))

  def count_non_matching_junk(self, s, s_matching_blocks, index_range=None):
    if not self.isjunk:
      return 0
    if index_range is None:
      index_range = (0, len(s))
    return sum(
      self.count_junk_between(s, block_index_range)
      for block_index_range in invert_index_ranges(
        s_matching_blocks, index_range[0], index_range[1]
      )
    )

  def a_junk_match_count(self):
    return self.count_junk_between(self.a, self.a_index_range())

  def a_junk_count(self):
    return self.count_junk_between(self.a, (0, len(self.a)))

  def a_non_matching_junk_count(self, index_range=None):
    return self.count_non_matching_junk(self.a, self.a_matching_blocks(), index_range)

  def b_junk_match_count(self):
    return self.count_junk_between(self.b, self.b_index_range())

  def b_junk_count(self):
    return self.count_junk_between(self.b, (0, len(self.b)))

  def b_non_matching_junk_count(self, index_range=None):
    return self.count_non_matching_junk(self.b, self.b_matching_blocks(), index_range)

  def a_ratio(self):
    return self.ratio_to(len(self.a) - self.a_non_matching_junk_count())

  def b_ratio(self):
    return self.ratio_to(len(self.b) - self.b_non_matching_junk_count())

  def b_gap_ratio(self):
    """
    Calculates the ratio of matches vs the length of b,
    but also adds any gaps / mismatches within a.
    """
    a_index_range = self.a_index_range()
    a_match_len = len_index_range(a_index_range)
    match_count = self.match_count()
    a_junk_match_count = self.a_non_matching_junk_count(a_index_range)
    b_junk_count = self.b_non_matching_junk_count()
    a_gaps = a_match_len - match_count
    return self.ratio_to(len(self.b) + a_gaps - a_junk_match_count - b_junk_count)

  def a_matching_blocks(self):
    return ((a, a + size) for a, _, size in self.non_empty_matching_blocks)

  def b_matching_blocks(self):
    return ((b, b + size) for _, b, size in self.non_empty_matching_blocks)

  def a_start_index(self):
    return self.non_empty_matching_blocks[0][0] if self.has_match() else None

  def a_end_index(self):
    if not self.has_match():
      return None
    ai, _, size = self.non_empty_matching_blocks[-1]
    return ai + size

  def a_index_range(self):
    if not self.non_empty_matching_blocks:
      return (0, 0)
    if not self._a_index_range:
      self._a_index_range = (self.a_start_index(), self.a_end_index())
    return self._a_index_range

  def b_start_index(self):
    return self.non_empty_matching_blocks[0][1] if self.has_match() else None

  def b_end_index(self):
    if not self.has_match():
      return None
    _, bi, size = self.non_empty_matching_blocks[-1]
    return bi + size

  def b_index_range(self):
    if not self.non_empty_matching_blocks:
      return (0, 0)
    if not self._b_index_range:
      self._b_index_range = (self.b_start_index(), self.b_end_index())
    return self._b_index_range

  def a_split_at(self, index, a_pre_split=None, a_post_split=None):
    if a_pre_split is None:
      a_pre_split = self.a[:index]
    if a_post_split is None:
      a_post_split = self.a[index:]
    if not self.non_empty_matching_blocks or self.a_end_index() <= index:
      return (
        FuzzyMatchResult(a_pre_split, self.b, self.non_empty_matching_blocks),
        FuzzyMatchResult(a_post_split, self.b, [])
      )
    return (
      FuzzyMatchResult(a_pre_split, self.b, [
        (ai, bi, min(size, index - ai))
        for ai, bi, size in self.non_empty_matching_blocks
        if ai < index
      ]),
      FuzzyMatchResult(a_post_split, self.b, [
        (max(0, ai - index), bi, size if ai >= index else size + ai - index)
        for ai, bi, size in self.non_empty_matching_blocks
        if ai + size > index
      ])
    )

  def b_split_at(self, index, b_pre_split=None, b_post_split=None):
    if b_pre_split is None:
      b_pre_split = self.b[:index]
    if b_post_split is None:
      b_post_split = self.b[index:]
    if not self.non_empty_matching_blocks or self.b_end_index() <= index:
      return (
        FuzzyMatchResult(self.a, b_pre_split, self.non_empty_matching_blocks),
        FuzzyMatchResult(self.a, b_post_split, [])
      )
    result = (
      FuzzyMatchResult(self.a, b_pre_split, [
        (ai, bi, min(size, index - bi))
        for ai, bi, size in self.non_empty_matching_blocks
        if bi < index
      ]),
      FuzzyMatchResult(self.a, b_post_split, [
        (ai, max(0, bi - index), size if bi >= index else size + bi - index)
        for ai, bi, size in self.non_empty_matching_blocks
        if bi + size > index
      ])
    )
    return result

  def detailed_str(self):
    return 'matching_blocks=[%s]' % (
      ', '.join([
        '(a[%d:+%d] = b[%d:+%d] = "%s")' % (ai, size, bi, size, self.a[ai:ai + size])
        for ai, bi, size in self.non_empty_matching_blocks
      ])
    )

  def detailed(self):
    return LazyStr(self.detailed_str)

  def __repr__(self):
    return (
      'FuzzyMatchResult(matching_blocks={}, match_count={}, ratio={},'
      ' a_ratio={}, b_gap_ratio={})'.format(
        self.matching_blocks, self.match_count(), self.ratio(), self.a_ratio(), self.b_gap_ratio()
      )
    )

def fuzzy_match(a, b, exact_word_match_threshold=5):
  if min(len(a), len(b)) < exact_word_match_threshold:
    sm = WordSequenceMatcher(None, a, b)
  else:
    sm = LocalSequenceMatcher(a=a, b=b, scoring=DEFAULT_SCORING)
  matching_blocks = sm.get_matching_blocks()
  return FuzzyMatchResult(a, b, matching_blocks)
