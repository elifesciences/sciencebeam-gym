from __future__ import division

import logging
import csv
from builtins import str as text
from itertools import tee, islice

from future.utils import python_2_unicode_compatible

from six.moves import zip_longest

from sciencebeam_gym.utils.csv import (
  csv_delimiter_by_filename,
  write_csv_row
)

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

from sciencebeam_gym.utils.collection import (
  iter_flatten,
  extract_from_dict
)

from sciencebeam_gym.preprocess.annotation.annotator import (
  AbstractAnnotator
)

THIN_SPACE = u'\u2009'
EN_DASH = u'\u2013'
EM_DASH = u'\u2014'

DEFAULT_SCORING = SimpleScoring(
  match_score=2,
  mismatch_score=-1,
  gap_score=-2
)

DEFAULT_SCORE_THRESHOLD = 0.9
DEFAULT_MAX_MATCH_GAP = 5

def get_logger():
  return logging.getLogger(__name__)

def normalise_str(s):
  return s.lower().replace(EM_DASH, u'-').replace(EN_DASH, u'-').replace(THIN_SPACE, ' ')

def normalise_str_or_list(x):
  if isinstance(x, list):
    return [normalise_str(s) for s in x]
  else:
    return normalise_str(x)

class SequenceWrapper(object):
  def __init__(self, structured_document, tokens, str_filter_f=None):
    self.structured_document = structured_document
    self.str_filter_f = str_filter_f
    self.tokens = tokens
    self.token_str_list = [structured_document.get_text(t) or '' for t in tokens]
    self.tokens_as_str = ' '.join(self.token_str_list)
    if str_filter_f:
      self.tokens_as_str = str_filter_f(self.tokens_as_str)

  def tokens_between(self, index_range):
    start, end = index_range
    i = 0
    for token, token_str in zip(self.tokens, self.token_str_list):
      if i >= end:
        break
      token_end = i + len(token_str)
      if token_end > start:
        yield token
      i = token_end + 1

  def sub_sequence_for_tokens(self, tokens):
    return SequenceWrapper(self.structured_document, tokens, str_filter_f=self.str_filter_f)

  def untagged_sub_sequences(self):
    token_tags = [self.structured_document.get_tag(t) for t in self.tokens]
    tagged_count = len([t for t in token_tags if t])
    if tagged_count == 0:
      yield self
    elif tagged_count == len(self.tokens):
      pass
    else:
      untagged_tokens = []
      for token, tag in zip(self.tokens, token_tags):
        if not tag:
          untagged_tokens.append(token)
        elif untagged_tokens:
          yield self.sub_sequence_for_tokens(untagged_tokens)
          untagged_tokens = []
      if untagged_tokens:
        yield self.sub_sequence_for_tokens(untagged_tokens)

  def __str__(self):
    return self.tokens_as_str

  def __repr__(self):
    return '{}({})'.format('SequenceWrapper', self.tokens_as_str)

class SequenceWrapperWithPosition(SequenceWrapper):
  def __init__(self, *args, **kwargs):
    position, kwargs = extract_from_dict(kwargs, 'position')
    super(SequenceWrapperWithPosition, self).__init__(*args, **kwargs)
    self.position = position

  def sub_sequence_for_tokens(self, tokens):
    return SequenceWrapperWithPosition(
      self.structured_document, tokens,
      str_filter_f=self.str_filter_f,
      position=self.position
    )

  def __repr__(self):
    return '{}({}, {})'.format('SequenceWrapperWithPosition', self.tokens_as_str, self.position)

@python_2_unicode_compatible
class SequenceMatch(object):
  def __init__(self, seq1, seq2, index1_range, index2_range):
    self.seq1 = seq1
    self.seq2 = seq2
    self.index1_range = index1_range
    self.index2_range = index2_range

  def __str__(self):
    return u"SequenceMatch('{}'[{}:{}], '{}'[{}:{}])".format(
      self.seq1,
      self.index1_range[0],
      self.index1_range[1],
      self.seq2,
      self.index2_range[0],
      self.index2_range[1]
    )

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
      yield i, min(end, r_end)
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
    return self.ratio_to(max_len_excl_junk)

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
    return ((a, size) for a, _, size in self.non_empty_matching_blocks)

  def b_matching_blocks(self):
    return ((b, size) for _, b, size in self.non_empty_matching_blocks)

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

  def __str__(self):
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

@python_2_unicode_compatible
class PositionedSequenceSet(object):
  def __init__(self):
    self.data = set()

  def add(self, sequence):
    self.data.add(sequence.position)

  def is_close_to_any(self, sequence, max_gap):
    if not max_gap or not self.data:
      return True
    position = sequence.position
    max_distance = max_gap + 1
    for other_position in self.data:
      if abs(position - other_position) <= max_distance:
        return True
    return False

  def __str__(self):
    return str(self.data)

def offset_range_by(index_range, offset):
  if not offset:
    return index_range
  return (offset + index_range[0], offset + index_range[1])

def skip_whitespaces(s, start):
  while start < len(s) and s[start].isspace():
    start += 1
  return start

def get_fuzzy_match_filter(
  b_score_threshold, min_match_count, total_match_threshold,
  ratio_min_match_count, ratio_threshold):
  def check(fm, fm_next=None):
    if (
      fm.match_count() >= ratio_min_match_count and
      fm.ratio() >= ratio_threshold):
      return True
    return (
      fm.b_gap_ratio() >= b_score_threshold and
      (
        (
          fm.match_count() >= min_match_count and
          (fm_next is None or fm_next.ratio() >= ratio_threshold)
        ) or
        fm.a_ratio() >= total_match_threshold
      )
    )
  return check

DEFAULT_SEQ_FUZZY_MATCH_FILTER = get_fuzzy_match_filter(
  DEFAULT_SCORE_THRESHOLD,
  5,
  0.9,
  50,
  0.9
)

DEFAULT_CHOICE_FUZZY_MATCH_FILTER = get_fuzzy_match_filter(
  DEFAULT_SCORE_THRESHOLD,
  1,
  0.9,
  100,
  0.9
)

class MatchDebugFields(object):
  TAG = 'tag'
  MATCH_MULTIPLE = 'match_multiple'
  TAG_VALUE_PRE = 'tag_value_pre'
  TAG_VALUE_CURRENT = 'tag_value_current'
  START_INDEX = 'start_index'
  NEXT_START_INDEX = 'next_start_index'
  REACHED_END = 'reached_end'
  CHOICE_COMBINED = 'choice_combined'
  CHOICE_CURRENT = 'choice_current'
  CHOICE_NEXT = 'choice_next'
  ACCEPTED = 'accepted'
  TAG_TO_CHOICE_MATCH = 'tag_to_choice_match'
  FM_COMBINED = 'fm_combined'
  FM_COMBINED_DETAILED = 'fm_combined_detailed'
  FM_CURRENT = 'fm_current'
  FM_CURRENT_DETAILED = 'fm_current_detailed'
  FM_NEXT = 'fm_next'
  FM_NEXT_DETAILED = 'fm_next_detailed'

DEFAULT_MATCH_DEBUG_COLUMNS = [
  MatchDebugFields.TAG,
  MatchDebugFields.MATCH_MULTIPLE,
  MatchDebugFields.TAG_VALUE_PRE,
  MatchDebugFields.TAG_VALUE_CURRENT,
  MatchDebugFields.START_INDEX,
  MatchDebugFields.NEXT_START_INDEX,
  MatchDebugFields.REACHED_END,
  MatchDebugFields.CHOICE_COMBINED,
  MatchDebugFields.CHOICE_CURRENT,
  MatchDebugFields.CHOICE_NEXT,
  MatchDebugFields.ACCEPTED,
  MatchDebugFields.TAG_TO_CHOICE_MATCH,
  MatchDebugFields.FM_COMBINED,
  MatchDebugFields.FM_COMBINED_DETAILED,
  MatchDebugFields.FM_CURRENT,
  MatchDebugFields.FM_CURRENT_DETAILED,
  MatchDebugFields.FM_NEXT,
  MatchDebugFields.FM_NEXT_DETAILED
]

def find_best_matches(
  target_annotation,
  sequence,
  choices,
  seq_match_filter=DEFAULT_SEQ_FUZZY_MATCH_FILTER,
  choice_match_filter=DEFAULT_CHOICE_FUZZY_MATCH_FILTER,
  max_gap=DEFAULT_MAX_MATCH_GAP,
  matched_choices=None,
  match_detail_reporter=None):

  if matched_choices is None:
    matched_choices = PositionedSequenceSet()
  if isinstance(sequence, list):
    get_logger().debug('found sequence list: %s', sequence)
    # Use tee as choices may be an iterable instead of a list
    for s, sub_choices in zip(sequence, tee(choices, len(sequence))):
      matches = find_best_matches(
        target_annotation,
        s,
        sub_choices,
        seq_match_filter=seq_match_filter,
        choice_match_filter=choice_match_filter,
        max_gap=max_gap,
        matched_choices=matched_choices,
        match_detail_reporter=match_detail_reporter
      )
      for m in matches:
        yield m
    return
  start_index = 0
  s1 = text(sequence)
  too_distant_choices = []

  current_choices, next_choices = tee(choices, 2)
  next_choices = islice(next_choices, 1, None)
  for choice, next_choice in zip_longest(current_choices, next_choices):
    if not matched_choices.is_close_to_any(choice, max_gap=max_gap):
      too_distant_choices.append(choice)
      continue
    current_choice_str = text(choice)
    if not current_choice_str:
      return
    if next_choice:
      next_choice_str = text(next_choice)
      choice_str = current_choice_str + ' ' + next_choice_str
    else:
      choice_str = current_choice_str
      next_choice_str = None
    current_start_index = start_index
    get_logger().debug(
      'processing choice: tag=%s, s1[:%d]=%s, s1[%d:]=%s, current=%s, next=%s (%s), combined=%s',
      target_annotation.name,
      start_index, s1[:start_index],
      start_index, s1[start_index:],
      current_choice_str,
      next_choice_str, type(next_choice_str), choice_str
    )
    fm_combined, fm, fm_next = None, None, None
    reached_end = None
    tag_to_choice_match = len(s1) - start_index < len(current_choice_str)
    if not tag_to_choice_match:
      fm_combined = fuzzy_match(s1, choice_str)
      fm, fm_next = fm_combined.b_split_at(len(current_choice_str))
      get_logger().debug(
        'regular match: s1=%s, choice=%s, fm=%s (combined: %s)',
        s1, choice, fm, fm_combined
      )
      get_logger().debug('detailed match: %s', fm_combined.detailed())
      accept_match = fm.has_match() and (
        seq_match_filter(fm, fm_next) or
        (seq_match_filter(fm_combined) and fm.b_start_index() < len(current_choice_str))
      )
      if accept_match:
        accept_match = True
        sm = SequenceMatch(
          sequence,
          choice,
          fm.a_index_range(),
          fm.b_index_range()
        )
        matched_choices.add(choice)
        get_logger().debug('found match: %s', sm)
        yield sm
        if fm_next.has_match():
          sm = SequenceMatch(
            sequence,
            next_choice,
            fm_next.a_index_range(),
            fm_next.b_index_range()
          )
          matched_choices.add(choice)
          get_logger().debug('found next match: %s', sm)
          yield sm
          index1_end = skip_whitespaces(s1, fm_next.a_end_index())
        else:
          index1_end = skip_whitespaces(s1, fm.a_end_index())
        reached_end = index1_end >= len(s1)
        if reached_end:
          get_logger().debug('end reached: %d >= %d', index1_end, len(s1))
          if target_annotation.match_multiple:
            start_index = 0
          else:
            break
        else:
          start_index = index1_end
          get_logger().debug('setting start index to: %d', start_index)
    else:
      s1_sub = s1[start_index:]
      fm_combined = fuzzy_match(choice_str, s1_sub)
      fm, fm_next = fm_combined.a_split_at(len(current_choice_str))
      get_logger().debug(
        'short match: s1_sub=%s, choice=%s, fm=%s (combined: %s)',
        s1_sub, choice, fm, fm_combined
      )
      get_logger().debug('detailed match: %s', fm_combined.detailed())
      accept_match = fm.has_match() and (
        choice_match_filter(fm) or
        (choice_match_filter(fm_combined) and fm_combined.a_start_index() < len(current_choice_str))
      )
      if accept_match:
        sm = SequenceMatch(
          sequence,
          choice,
          offset_range_by(fm.b_index_range(), start_index),
          fm.a_index_range()
        )
        matched_choices.add(choice)
        get_logger().debug('found match: %s', sm)
        yield sm
        if fm_next.has_match():
          sm = SequenceMatch(
            sequence,
            next_choice,
            offset_range_by(fm_next.b_index_range(), start_index),
            fm_next.a_index_range()
          )
          get_logger().debug('found next match: %s', sm)
          matched_choices.add(next_choice)
          yield sm
        if not target_annotation.match_multiple:
          break
    if match_detail_reporter:
      match_detail_reporter({
        MatchDebugFields.TAG: target_annotation.name,
        MatchDebugFields.MATCH_MULTIPLE: target_annotation.match_multiple,
        MatchDebugFields.TAG_VALUE_PRE: s1[:current_start_index],
        MatchDebugFields.TAG_VALUE_CURRENT: s1[current_start_index:],
        MatchDebugFields.START_INDEX: current_start_index,
        MatchDebugFields.NEXT_START_INDEX: start_index,
        MatchDebugFields.REACHED_END: reached_end,
        MatchDebugFields.CHOICE_COMBINED: choice_str,
        MatchDebugFields.CHOICE_CURRENT: current_choice_str,
        MatchDebugFields.CHOICE_NEXT: next_choice_str,
        MatchDebugFields.ACCEPTED: accept_match,
        MatchDebugFields.TAG_TO_CHOICE_MATCH: tag_to_choice_match,
        MatchDebugFields.FM_COMBINED: fm_combined,
        MatchDebugFields.FM_COMBINED_DETAILED: fm_combined and fm_combined.detailed_str(),
        MatchDebugFields.FM_CURRENT: fm,
        MatchDebugFields.FM_CURRENT_DETAILED: fm and fm.detailed_str(),
        MatchDebugFields.FM_NEXT: fm_next,
        MatchDebugFields.FM_NEXT_DETAILED: fm_next.detailed_str()
      })
  if too_distant_choices:
    get_logger().debug(
      'ignored too distant choices: matched=%s (ignored=%s)',
      matched_choices,
      LazyStr(lambda: ' '.join(str(choice.position) for choice in too_distant_choices))
    )

class CsvMatchDetailReporter(object):
  def __init__(self, fp, filename=None, fields=None):
    self.fp = fp
    self.fields = fields or DEFAULT_MATCH_DEBUG_COLUMNS
    self.writer = csv.writer(
      fp,
      delimiter=csv_delimiter_by_filename(filename)
    )
    self.writer.writerow(self.fields)

  def __call__(self, row):
    write_csv_row(self.writer, [row.get(k) for k in self.fields])

  def close(self):
    self.fp.close()

class MatchingAnnotator(AbstractAnnotator):
  def __init__(self, target_annotations, match_detail_reporter=None):
    self.target_annotations = target_annotations
    self.match_detail_reporter = match_detail_reporter

  def annotate(self, structured_document):
    pending_sequences = []
    for page in structured_document.get_pages():
      for line in structured_document.get_lines_of_page(page):
        tokens = [
          token
          for token in structured_document.get_tokens_of_line(line)
          if not structured_document.get_tag(token)
        ]
        if tokens:
          get_logger().debug(
            'tokens without tag: %s',
            [structured_document.get_text(token) for token in tokens]
          )
          pending_sequences.append(SequenceWrapperWithPosition(
            structured_document,
            tokens,
            normalise_str,
            position=len(pending_sequences)
          ))

    matched_choices_map = dict()
    for target_annotation in self.target_annotations:
      get_logger().debug('target annotation: %s', target_annotation)
      target_value = normalise_str_or_list(target_annotation.value)
      untagged_pending_sequences = iter_flatten(
        seq.untagged_sub_sequences() for seq in pending_sequences
      )
      if target_annotation.bonding:
        matched_choices = matched_choices_map.setdefault(
          target_annotation.name,
          PositionedSequenceSet()
        )
      else:
        matched_choices = PositionedSequenceSet()
      matches = find_best_matches(
        target_annotation,
        target_value,
        untagged_pending_sequences,
        matched_choices=matched_choices,
        match_detail_reporter=self.match_detail_reporter
      )
      for m in matches:
        choice = m.seq2
        matching_tokens = list(choice.tokens_between(m.index2_range))
        get_logger().debug(
          'matching_tokens: %s %s',
          [structured_document.get_text(token) for token in matching_tokens],
          m.index2_range
        )
        for token in matching_tokens:
          if not structured_document.get_tag(token):
            structured_document.set_tag(
              token,
              target_annotation.name
            )
    return structured_document
