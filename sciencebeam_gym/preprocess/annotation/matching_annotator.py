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

from sciencebeam_gym.utils.collection import (
  iter_flatten,
  extract_from_dict
)

from sciencebeam_gym.structured_document import (
  B_TAG_PREFIX,
  I_TAG_PREFIX
)

from sciencebeam_gym.preprocess.annotation.fuzzy_match import (
  fuzzy_match
)

from sciencebeam_gym.preprocess.annotation.annotator import (
  AbstractAnnotator
)

THIN_SPACE = u'\u2009'
EN_DASH = u'\u2013'
EM_DASH = u'\u2014'

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

  def __repr__(self):
    return u"SequenceMatch('{}'[{}:{}], '{}'[{}:{}])".format(
      self.seq1,
      self.index1_range[0],
      self.index1_range[1],
      self.seq2,
      self.index2_range[0],
      self.index2_range[1]
    )

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
  def check(fm, fm_next=None, previous_match=False):
    if (
      fm.match_count() >= ratio_min_match_count and
      fm.ratio() >= ratio_threshold):
      return True
    return (
      fm.b_gap_ratio() >= b_score_threshold and
      (
        previous_match or
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
  ID = 'id'
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
  SUB_ANNOTATION = 'sub_annotation'
  FM_COMBINED = 'fm_combined'
  FM_COMBINED_DETAILED = 'fm_combined_detailed'
  FM_CURRENT = 'fm_current'
  FM_CURRENT_DETAILED = 'fm_current_detailed'
  FM_NEXT = 'fm_next'
  FM_NEXT_DETAILED = 'fm_next_detailed'

DEFAULT_MATCH_DEBUG_COLUMNS = [
  MatchDebugFields.ID,
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
  MatchDebugFields.SUB_ANNOTATION,
  MatchDebugFields.FM_COMBINED,
  MatchDebugFields.FM_COMBINED_DETAILED,
  MatchDebugFields.FM_CURRENT,
  MatchDebugFields.FM_CURRENT_DETAILED,
  MatchDebugFields.FM_NEXT,
  MatchDebugFields.FM_NEXT_DETAILED
]

class TargetAnnotationMatchFinder(object):
  def __init__(
    self,
    target_annotation,
    sequence,
    choices,
    seq_match_filter=DEFAULT_SEQ_FUZZY_MATCH_FILTER,
    choice_match_filter=DEFAULT_CHOICE_FUZZY_MATCH_FILTER,
    max_gap=DEFAULT_MAX_MATCH_GAP,
    matched_choices=None,
    match_detail_reporter=None,
    is_sub_match=False
  ):
    if matched_choices is None:
      matched_choices = PositionedSequenceSet()
    self.target_annotation = target_annotation
    self.sequence = sequence
    self.choices = choices
    self.seq_match_filter = seq_match_filter
    self.choice_match_filter = choice_match_filter
    self.max_gap = max_gap
    self.matched_choices = matched_choices
    self.match_detail_reporter = match_detail_reporter
    self.is_sub_match = is_sub_match
    self.current_choices, self.next_choices = tee(choices, 2)
    self.next_choices = islice(self.next_choices, 1, None)

  def find_next_best_matches(self):
    if isinstance(self.sequence, list):
      all_matches = []
      get_logger().debug('found sequence list: %s', self.sequence)
      # Use tee as choices may be an iterable instead of a list
      for s, sub_current_choices, sub_next_choices in zip(
        self.sequence,
        tee(self.current_choices, len(self.sequence)),
        tee(self.next_choices, len(self.sequence))
      ):
        all_matches.extend(self._do_find_next_best_matches(
          s,
          sub_current_choices,
          sub_next_choices
        ))
      get_logger().debug(
        'all_matches (bonding=%s): %s', self.target_annotation.bonding, all_matches
      )
      if not self.target_annotation.bonding or len(all_matches) > 1:
        for m in all_matches:
          yield m
    else:
      matches = self._do_find_next_best_matches(
        self.sequence, self.current_choices, self.next_choices
      )
      for m in matches:
        yield m

  def _do_find_next_best_matches(self, sequence, current_choices, next_choices):
    target_annotation = self.target_annotation
    seq_match_filter = self.seq_match_filter
    choice_match_filter = self.choice_match_filter
    max_gap = self.max_gap
    matched_choices = self.matched_choices

    start_index = 0
    s1 = text(sequence)
    too_distant_choices = []

    is_last_match = False
    previous_match = False

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
      tag_to_choice_match = self.is_sub_match or (len(s1) - start_index < len(current_choice_str))
      if not tag_to_choice_match:
        fm_combined = fuzzy_match(s1, choice_str)
        fm, fm_next = fm_combined.b_split_at(len(current_choice_str))
        get_logger().debug(
          'regular match: s1=%s, choice=%s, fm=%s (combined: %s)',
          s1, choice, fm, fm_combined
        )
        get_logger().debug('detailed match: %s', fm_combined.detailed())
        accept_match = fm.has_match() and (
          seq_match_filter(fm, fm_next, previous_match=previous_match) or
          (seq_match_filter(fm_combined) and fm.b_start_index() < len(current_choice_str))
        )
        if accept_match:
          previous_match = True
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
            is_last_match = True
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
          choice_match_filter(fm, previous_match=previous_match) or
          (
            choice_match_filter(fm_combined) and
            fm_combined.a_start_index() < len(current_choice_str)
          )
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
          is_last_match = True
      if self.match_detail_reporter:
        self.match_detail_reporter({
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
          MatchDebugFields.SUB_ANNOTATION: self.is_sub_match,
          MatchDebugFields.FM_COMBINED: fm_combined,
          MatchDebugFields.FM_COMBINED_DETAILED: fm_combined and fm_combined.detailed_str(),
          MatchDebugFields.FM_CURRENT: fm,
          MatchDebugFields.FM_CURRENT_DETAILED: fm and fm.detailed_str(),
          MatchDebugFields.FM_NEXT: fm_next,
          MatchDebugFields.FM_NEXT_DETAILED: fm_next.detailed_str()
        })
      if is_last_match:
        break
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
    self.id = 1

  def __call__(self, row):
    get_logger().debug('logging debug match id %d', self.id)
    write_csv_row(self.writer, [
      self.id if k == MatchDebugFields.ID else row.get(k)
      for k in self.fields
    ])
    self.id += 1

  def close(self):
    self.fp.close()

def sorted_matches_by_position(matches):
  return sorted(
    matches,
    key=lambda m: (m.seq2.position, m.index2_range)
  )

def _apply_sub_annotations(
  target_annotation, structured_document, matching_tokens,
  match_detail_reporter, use_tag_begin_prefix):

  seq = SequenceWrapperWithPosition(
    structured_document, matching_tokens, normalise_str,
    position=0
  )
  matched_choices = PositionedSequenceSet()
  for sub_annotation in target_annotation.sub_annotations:
    sub_tag = sub_annotation.name
    match_finder = TargetAnnotationMatchFinder(
      sub_annotation,
      normalise_str_or_list(sub_annotation.value),
      [seq],
      matched_choices=matched_choices,
      match_detail_reporter=match_detail_reporter,
      is_sub_match=True
    )
    matches = match_finder.find_next_best_matches()
    matches = list(matches)
    get_logger().info('sub annotation matches: %s', matches)
    first_token = True
    for m in matches:
      choice = m.seq2
      matching_sub_tokens = list(choice.tokens_between(m.index2_range))
      get_logger().debug(
        'matching_sub_tokens: %s %s',
        [structured_document.get_text(token) for token in matching_tokens],
        m.index2_range
      )
      for token in matching_sub_tokens:
        tag_prefix = None
        if use_tag_begin_prefix:
          tag_prefix = B_TAG_PREFIX if first_token else I_TAG_PREFIX
        structured_document.set_sub_tag_with_prefix(token, sub_tag, prefix=tag_prefix)
        first_token = False

def _apply_annotations_to_matches(
  target_annotation,
  structured_document,
  matches,
  match_detail_reporter,
  use_tag_begin_prefix):

  first_token = True
  all_matching_tokens = []
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
        tag_prefix = None
        if use_tag_begin_prefix:
          tag_prefix = B_TAG_PREFIX if first_token else I_TAG_PREFIX
        structured_document.set_tag_with_prefix(
          token,
          target_annotation.name,
          prefix=tag_prefix
        )
        first_token = False
        all_matching_tokens.append(token)
    if target_annotation.sub_annotations:
      _apply_sub_annotations(
        target_annotation, structured_document, all_matching_tokens,
        match_detail_reporter, use_tag_begin_prefix
      )


class MatchingAnnotator(AbstractAnnotator):
  def __init__(
    self, target_annotations, match_detail_reporter=None,
    use_tag_begin_prefix=False):

    self.target_annotations = target_annotations
    self.match_detail_reporter = match_detail_reporter
    self.use_tag_begin_prefix = use_tag_begin_prefix

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

    conditional_match = None

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
      match_finder = TargetAnnotationMatchFinder(
        target_annotation,
        target_value,
        untagged_pending_sequences,
        matched_choices=matched_choices,
        match_detail_reporter=self.match_detail_reporter
      )
      item_index = 0
      while item_index == 0 or target_annotation.match_multiple:
        get_logger().info('calling find_next_best_matches')
        matches = sorted_matches_by_position(
          match_finder.find_next_best_matches()
        )
        if not matches:
          conditional_match = None
          break
        get_logger().info('matches: %s', matches)
        if conditional_match:
          _apply_annotations_to_matches(
            conditional_match['target_annotation'],
            structured_document,
            conditional_match['matches'],
            self.match_detail_reporter, self.use_tag_begin_prefix
          )
        if target_annotation.require_next:
          conditional_match = dict(
            target_annotation=target_annotation,
            matches=matches
          )
        else:
          _apply_annotations_to_matches(
            target_annotation, structured_document, matches,
            self.match_detail_reporter, self.use_tag_begin_prefix
          )
        item_index += 1
    return structured_document
