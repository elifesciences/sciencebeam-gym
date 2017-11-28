from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from six import (
  with_metaclass,
  u as as_u,
  b as as_b
)

import numpy as np

from sciencebeam_gym.alignment.align import (
  LocalSequenceMatcher,
  GlobalSequenceMatcher,
  SimpleScoring,
  CustomScoring,
  require_native
)

DEFAULT_SCORING = SimpleScoring(
  match_score=3,
  mismatch_score=-1,
  gap_score=-2
)

DEFAULT_CUSTOM_SCORING = CustomScoring(
  scoring_fn=lambda a, b: (
    DEFAULT_SCORING.match_score if a == b else DEFAULT_SCORING.mismatch_score
  ),
  gap_score=-2
)

def _non_zero(matching_blocks):
  return [(ai, bi, size) for ai, bi, size in matching_blocks if size]

class CharWrapper(object):
  def __init__(self, c):
    self.c = c

  def __eq__(self, b):
    return self.c == b.c

class AbstractTestCommonSequenceMatcher(object, with_metaclass(ABCMeta)):
  @abstractmethod
  def _convert(self, x):
    pass

  @contextmanager
  def _wrap_test(self):
    yield

  def test_should_not_return_non_zero_blocks_for_no(self):
    with self._wrap_test():
      sm = self._matcher(a='a', b='b')
      assert _non_zero(sm.get_matching_blocks()) == []

  def test_should_add_zero_block_with_final_indices(self):
    with self._wrap_test():
      sm = self._matcher(a='a', b='b')
      assert sm.get_matching_blocks() == [(1, 1, 0)]

  def test_should_return_single_character_match(self):
    with self._wrap_test():
      sm = self._matcher(a='a', b='a')
      assert _non_zero(sm.get_matching_blocks()) == [(0, 0, 1)]

  def test_should_return_combine_character_match_block(self):
    with self._wrap_test():
      sm = self._matcher(a='abc', b='abc')
      assert _non_zero(sm.get_matching_blocks()) == [(0, 0, 3)]

  def test_should_return_combine_character_match_blocks_with_gaps(self):
    with self._wrap_test():
      sm = self._matcher(a='abc123xyz', b='abc987xyz')
      assert _non_zero(sm.get_matching_blocks()) == [(0, 0, 3), (6, 6, 3)]

  def test_should_align_using_custom_scoring_fn(self):
    with self._wrap_test():
      sm = self._matcher(a='a', b='a', scoring=DEFAULT_CUSTOM_SCORING)
      assert _non_zero(sm.get_matching_blocks()) == [(0, 0, 1)]

class AbstractTestLocalSequenceMatcher(AbstractTestCommonSequenceMatcher):
  def _matcher(self, a, b, scoring=None):
    return LocalSequenceMatcher(
      a=self._convert(a),
      b=self._convert(b),
      scoring=scoring or DEFAULT_SCORING
    )

  def test_should_not_match_block_after_big_gap(self):
    with self._wrap_test():
      sm = self._matcher(a='abcxyz', b='abc123456xyz')
      assert _non_zero(sm.get_matching_blocks()) == [(0, 0, 3)]

class TestLocalSequenceMatcherWithUnicode(AbstractTestLocalSequenceMatcher):
  def _convert(self, x):
    return as_u(x)

class TestLocalSequenceMatcherWithBytes(AbstractTestLocalSequenceMatcher):
  def _convert(self, x):
    return as_b(x)

class TestLocalSequenceMatcherWithIntList(AbstractTestLocalSequenceMatcher):
  def _convert(self, x):
    return [ord(c) for c in x]

class TestLocalSequenceMatcherWithNumpyInt32Array(AbstractTestLocalSequenceMatcher):
  def _convert(self, x):
    return np.array([ord(c) for c in x], dtype=np.int32)

class TestLocalSequenceMatcherWithNumpyInt64Array(AbstractTestLocalSequenceMatcher):
  def _convert(self, x):
    return np.array([ord(c) for c in x], dtype=np.int64)

class TestLocalSequenceMatcherWithCustomObjectList(AbstractTestLocalSequenceMatcher):
  def _convert(self, x):
    return [CharWrapper(c) for c in x]

class TestLocalSequenceMatcherWithNumpyInt32ArrayWithoutNative(
  TestLocalSequenceMatcherWithNumpyInt32Array):

  @contextmanager
  def _wrap_test(self):
    with require_native(False):
      yield

class AbstractTestGlobalSequenceMatcher(AbstractTestCommonSequenceMatcher):
  def _matcher(self, a, b, scoring=None):
    with require_native(False):
      return GlobalSequenceMatcher(
        a=self._convert(a),
        b=self._convert(b),
        scoring=scoring or DEFAULT_SCORING
      )

  def test_should_prefer_match_block_after_big_gap(self):
    with self._wrap_test():
      sm = self._matcher(a='abcxyz', b='abc123456xyz')
      assert _non_zero(sm.get_matching_blocks()) == [(0, 0, 3), (3, 9, 3)]

class TestGlobalSequenceMatcherWithUnicode(AbstractTestGlobalSequenceMatcher):
  def _convert(self, x):
    return as_u(x)
