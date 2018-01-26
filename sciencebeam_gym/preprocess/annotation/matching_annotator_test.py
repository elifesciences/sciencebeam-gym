from __future__ import division

import logging

from sciencebeam_gym.structured_document import (
  SimpleStructuredDocument,
  SimpleLine,
  SimpleToken
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
  TargetAnnotation
)

from sciencebeam_gym.preprocess.annotation.matching_annotator import (
  normalise_str,
  MatchingAnnotator,
  THIN_SPACE,
  EN_DASH,
  EM_DASH
)

from sciencebeam_gym.utils.collection import (
  flatten
)

TAG1 = 'tag1'
TAG2 = 'tag2'

B_TAG_1 = 'b-' + TAG1
I_TAG_1 = 'i-' + TAG1

B_TAG_2 = 'b-' + TAG2
I_TAG_2 = 'i-' + TAG2

SOME_VALUE = 'some value'
SOME_VALUE_2 = 'some value2'
SOME_LONGER_VALUE = 'some longer value1'
SOME_SHORTER_VALUE = 'value1'

def setup_module():
  logging.basicConfig(level='DEBUG')

def _get_tags_of_tokens(tokens):
  return [t.get_tag() for t in tokens]

def _copy_tokens(tokens):
  return [SimpleToken(t.text) for t in tokens]

def _tokens_for_text(text):
  return [SimpleToken(s) for s in text.split(' ')]

def _tokens_to_text(tokens):
  return ' '.join([t.text for t in tokens])

def _tokens_for_text_lines(text_lines):
  return [_tokens_for_text(line) for line in text_lines]

def _lines_for_tokens(tokens_by_line):
  return [SimpleLine(tokens) for tokens in tokens_by_line]

def _document_for_tokens(tokens_by_line):
  return SimpleStructuredDocument(lines=_lines_for_tokens(tokens_by_line))

class TestNormaliseStr(object):
  def test_should_replace_thin_space_with_regular_space(self):
    assert normalise_str(THIN_SPACE) == ' '

  def test_should_replace_em_dash_with_hyphen(self):
    assert normalise_str(EM_DASH) == '-'

  def test_should_replace_en_dash_with_hyphen(self):
    assert normalise_str(EN_DASH) == '-'

class TestMatchingAnnotator(object):
  def test_should_not_fail_on_empty_document(self):
    doc = SimpleStructuredDocument(lines=[])
    MatchingAnnotator([]).annotate(doc)

  def test_should_not_fail_on_empty_line_with_blank_token(self):
    target_annotations = [
      TargetAnnotation('this is. matching', TAG1)
    ]
    doc = _document_for_tokens([[SimpleToken('')]])
    MatchingAnnotator(target_annotations).annotate(doc)

  def test_should_annotate_exactly_matching(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is matching', TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_use_begin_prefix_if_enabled(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is matching', TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=True).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [B_TAG_1, I_TAG_1, I_TAG_1]

  def test_should_match_normalised_characters(self):
    matching_tokens = [
      SimpleToken('this'),
      SimpleToken('is' + THIN_SPACE + EN_DASH + EM_DASH),
      SimpleToken('matching')
    ]
    target_annotations = [
      TargetAnnotation('this is -- matching', TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_match_case_insensitive(self):
    matching_tokens = _tokens_for_text('This Is Matching')
    target_annotations = [
      TargetAnnotation('tHIS iS mATCHING', TAG1)
    ]
    doc = SimpleStructuredDocument(lines=[SimpleLine(matching_tokens)])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_prefer_word_boundaries(self):
    pre_tokens = _tokens_for_text('this')
    matching_tokens = _tokens_for_text('is')
    post_tokens = _tokens_for_text('miss')
    target_annotations = [
      TargetAnnotation('is', TAG1)
    ]
    doc = _document_for_tokens([
      pre_tokens + matching_tokens + post_tokens
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(pre_tokens) == [None] * len(pre_tokens)
    assert _get_tags_of_tokens(post_tokens) == [None] * len(post_tokens)

  def test_should_annotate_multiple_value_target_annotation(self):
    matching_tokens = _tokens_for_text('this may match')
    target_annotations = [
      TargetAnnotation([
        'this', 'may', 'match'
      ], TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_annotate_multiple_value_target_annotation_with_begin_prefix(self):
    matching_tokens = _tokens_for_text('this may match')
    target_annotations = [
      TargetAnnotation([
        'this', 'may', 'match'
      ], TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=True).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == (
      [B_TAG_1] + [I_TAG_1] * (len(matching_tokens) - 1)
    )

  def test_should_annotate_multiple_value_target_annotation_rev_order_with_begin_prefix(self):
    matching_tokens = _tokens_for_text('this may match')
    target_annotations = [
      TargetAnnotation(list(reversed([
        'this', 'may', 'match'
      ])), TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=True).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == (
      [B_TAG_1] + [I_TAG_1] * (len(matching_tokens) - 1)
    )

  def test_should_annotate_multiple_value_target_annotation_over_multiple_lines(self):
    tokens_by_line = [
      _tokens_for_text('this may'),
      _tokens_for_text('match')
    ]
    matching_tokens = flatten(tokens_by_line)
    target_annotations = [
      TargetAnnotation([
        'this', 'may', 'match'
      ], TAG1)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_annotate_mult_value_target_annot_rev_order_over_mult_lines_with_b_prefix(self):
    tokens_by_line = [
      _tokens_for_text('this may'),
      _tokens_for_text('match')
    ]
    matching_tokens = flatten(tokens_by_line)
    target_annotations = [
      TargetAnnotation(list(reversed([
        'this', 'may', 'match'
      ])), TAG1)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=True).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == (
      [B_TAG_1] +
      [I_TAG_1] * (len(matching_tokens) - 1)
    )

  def test_should_annotate_not_match_distant_value_of_multiple_value_target_annotation(self):
    matching_tokens = _tokens_for_text('this may match')
    distant_matching_tokens = _tokens_for_text('not')
    distance_in_lines = 10
    tokens_by_line = [matching_tokens] + [
      _tokens_for_text('other') for _ in range(distance_in_lines)
    ] + [distant_matching_tokens]
    target_annotations = [
      TargetAnnotation([
        'this', 'may', 'match', 'not'
      ], TAG1)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(distant_matching_tokens) == [None] * len(distant_matching_tokens)

  def test_should_annotate_not_match_distant_value_of_target_annotation_with_bonding(self):
    matching_tokens = _tokens_for_text('this may match')
    distant_matching_tokens = _tokens_for_text('not')
    distance_in_lines = 10
    tokens_by_line = [matching_tokens] + [
      _tokens_for_text('other') for _ in range(distance_in_lines)
    ] + [distant_matching_tokens]
    target_annotations = [
      TargetAnnotation('this may match', TAG1, bonding=True),
      TargetAnnotation('not', TAG1, bonding=True)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(distant_matching_tokens) == [None] * len(distant_matching_tokens)

  def test_should_annotate_fuzzily_matching(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is. matching', TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_annotate_ignoring_space_after_dot_short_sequence(self):
    matching_tokens = [
      SimpleToken('A.B.,')
    ]
    target_annotations = [
      TargetAnnotation('A. B.', TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_annotate_ignoring_comma_after_short_sequence(self):
    matching_tokens = [
      SimpleToken('Name,'),
    ]
    target_annotations = [
      TargetAnnotation('Name', TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_annotate_with_local_matching_smaller_gaps(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is. matching indeed matching', TAG1)
    ]
    # this should align with 'this is_ matching' with one gap'
    # instead of globally 'this is_ ________ ______ matching'
    # (which would result in a worse b_gap_ratio)
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_not_annotate_fuzzily_matching_with_many_differences(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('txhxixsx ixsx mxaxtxcxhxixnxgx', TAG1)
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [None] * len(matching_tokens)

  def test_should_annotate_fuzzily_matching_longer_matches_based_on_ratio(self):
    long_matching_text = 'this is matching and is really really long match that we can trust'
    matching_tokens = _tokens_for_text(long_matching_text)
    no_matching_tokens = _tokens_for_text('what comes next is different')
    target_annotations = [
      TargetAnnotation(long_matching_text + ' but this is not and is another matter', TAG1)
    ]
    doc = _document_for_tokens([
      matching_tokens + no_matching_tokens
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(no_matching_tokens) == [None] * len(no_matching_tokens)

  def test_should_not_annotate_not_matching(self):
    not_matching_tokens = _tokens_for_text('something completely different')
    target_annotations = [
      TargetAnnotation('this is matching', TAG1)
    ]
    doc = _document_for_tokens([not_matching_tokens])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(not_matching_tokens) == [None] * len(not_matching_tokens)

  def test_should_annotate_exactly_matching_across_multiple_lines(self):
    matching_tokens_per_line = [
      _tokens_for_text('this is matching'),
      _tokens_for_text('and continues here')
    ]
    matching_tokens = flatten(matching_tokens_per_line)
    target_annotations = [
      TargetAnnotation('this is matching and continues here', TAG1)
    ]
    doc = _document_for_tokens(matching_tokens_per_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_annotate_exactly_matching_across_multiple_lines_with_begin_prefix(self):
    matching_tokens_per_line = [
      _tokens_for_text('this is matching'),
      _tokens_for_text('and continues here')
    ]
    matching_tokens = flatten(matching_tokens_per_line)
    target_annotations = [
      TargetAnnotation('this is matching and continues here', TAG1)
    ]
    doc = _document_for_tokens(matching_tokens_per_line)
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=True).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == (
      [B_TAG_1] + [I_TAG_1] * (len(matching_tokens) - 1)
    )

  def test_should_not_annotate_shorter_sequence_if_next_line_does_not_match(self):
    tokens_per_line = [
      _tokens_for_text('this is'),
      _tokens_for_text('something completely different')
    ]
    tokens = flatten(tokens_per_line)
    target_annotations = [
      TargetAnnotation('this is not matching', TAG1)
    ]
    doc = _document_for_tokens(tokens_per_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(tokens) == [None] * len(tokens)

  def test_should_annotate_over_multiple_lines_with_tag_transition(self):
    tag1_tokens_by_line = [
      _tokens_for_text('this may'),
      _tokens_for_text('match')
    ]
    tag1_tokens = flatten(tag1_tokens_by_line)
    tag2_tokens_by_line = [
      _tokens_for_text('another'),
      _tokens_for_text('tag here')
    ]
    tag2_tokens = flatten(tag2_tokens_by_line)
    tokens_by_line = [
      tag1_tokens_by_line[0],
      tag1_tokens_by_line[1] + tag2_tokens_by_line[0],
      tag2_tokens_by_line[1]
    ]
    target_annotations = [
      TargetAnnotation('this may match', TAG1),
      TargetAnnotation('another tag here', TAG2)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(tag1_tokens) == [TAG1] * len(tag1_tokens)
    assert _get_tags_of_tokens(tag2_tokens) == [TAG2] * len(tag2_tokens)

  def test_should_annotate_over_multiple_lines_with_tag_transition_with_begin_prefix(self):
    tag1_tokens_by_line = [
      _tokens_for_text('this may'),
      _tokens_for_text('match')
    ]
    tag1_tokens = flatten(tag1_tokens_by_line)
    tag2_tokens_by_line = [
      _tokens_for_text('another'),
      _tokens_for_text('tag here')
    ]
    tag2_tokens = flatten(tag2_tokens_by_line)
    tokens_by_line = [
      tag1_tokens_by_line[0],
      tag1_tokens_by_line[1] + tag2_tokens_by_line[0],
      tag2_tokens_by_line[1]
    ]
    target_annotations = [
      TargetAnnotation('this may match', TAG1),
      TargetAnnotation('another tag here', TAG2)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=True).annotate(doc)
    assert _get_tags_of_tokens(tag1_tokens) == (
      [B_TAG_1] + [I_TAG_1] * (len(tag1_tokens) - 1)
    )
    assert _get_tags_of_tokens(tag2_tokens) == (
      [B_TAG_2] + [I_TAG_2] * (len(tag2_tokens) - 1)
    )

  def test_should_annotate_longer_sequence_over_multiple_lines_considering_next_line(self):
    # we need a long enough sequence to fall into the first branch
    # and match the partial match threshold
    exact_matching_text_lines = ('this may', 'indeed match very well without the slightest doubt')
    # add a short prefix that doesn't affect the score much
    # but would be skipped if we only matched the second line
    matching_text_lines = (exact_matching_text_lines[0], 'x ' + exact_matching_text_lines[1])
    matching_tokens_by_line = _tokens_for_text_lines(matching_text_lines)
    matching_tokens = flatten(matching_tokens_by_line)
    pre_tokens = _tokens_for_text(matching_text_lines[0] + ' this may not')
    post_tokens = _tokens_for_text('or not')
    tokens_by_line = [
      pre_tokens + matching_tokens_by_line[0],
      matching_tokens_by_line[1] + post_tokens
    ]
    target_annotations = [
      TargetAnnotation(' '.join(exact_matching_text_lines), TAG1)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(pre_tokens) == [None] * len(pre_tokens)
    assert _get_tags_of_tokens(post_tokens) == [None] * len(post_tokens)

  def test_should_annotate_shorter_sequence_over_multiple_lines_considering_next_line(self):
    # use a short sequence that wouldn't get matched on it's own
    matching_text_lines = ('this may', 'match')
    matching_tokens_by_line = _tokens_for_text_lines(matching_text_lines)
    matching_tokens = flatten(matching_tokens_by_line)
    # repeat the same text on the two lines, only by combining the lines would it be clear
    # which tokens to match
    pre_tokens = _tokens_for_text(matching_text_lines[0] + ' be some other longer preceeding text')
    post_tokens = _tokens_for_text('this is some text after but no ' + matching_text_lines[1])
    tokens_by_line = [
      pre_tokens + matching_tokens_by_line[0],
      matching_tokens_by_line[1] + post_tokens
    ]
    target_annotations = [
      TargetAnnotation('this may match', TAG1)
    ]
    doc = _document_for_tokens(tokens_by_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(pre_tokens) == [None] * len(pre_tokens)
    assert _get_tags_of_tokens(post_tokens) == [None] * len(post_tokens)

  def test_should_not_annotate_too_short_match_of_longer_sequence(self):
    matching_tokens = _tokens_for_text('this is matching')
    too_short_tokens = _tokens_for_text('1')
    tokens_per_line = [
      too_short_tokens,
      matching_tokens
    ]
    target_annotations = [
      TargetAnnotation('this is matching 1', TAG1)
    ]
    doc = _document_for_tokens(tokens_per_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(too_short_tokens) == [None] * len(too_short_tokens)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_not_annotate_similar_sequence_multiple_times(self):
    matching_tokens_per_line = [
      _tokens_for_text('this is matching'),
      _tokens_for_text('and continues here')
    ]
    not_matching_tokens = _tokens_for_text('this is matching')

    matching_tokens = flatten(matching_tokens_per_line)
    target_annotations = [
      TargetAnnotation('this is matching and continues here', TAG1)
    ]
    doc = _document_for_tokens(
      matching_tokens_per_line + [not_matching_tokens]
    )
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(not_matching_tokens) == [None] * len(not_matching_tokens)

  def test_should_annotate_same_sequence_multiple_times_if_enabled(self):
    matching_tokens_per_line = [
      _tokens_for_text('this is matching'),
      _tokens_for_text('this is matching')
    ]

    matching_tokens = flatten(matching_tokens_per_line)
    target_annotations = [
      TargetAnnotation('this is matching', TAG1, match_multiple=True)
    ]
    doc = _document_for_tokens(matching_tokens_per_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_not_override_annotation(self):
    matching_tokens_per_line = [
      _tokens_for_text('this is matching')
    ]

    matching_tokens = flatten(matching_tokens_per_line)
    target_annotations = [
      TargetAnnotation('this is matching', TAG1),
      TargetAnnotation('matching', TAG2)
    ]
    doc = _document_for_tokens(matching_tokens_per_line)
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_not_annotate_pre_annotated_tokens_on_separate_lines(self):
    line_no_tokens = _tokens_for_text('1')
    line_no_tokens[0].set_tag('line_no')
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('1', TAG2),
      TargetAnnotation('this is matching', TAG1)
    ]
    doc = _document_for_tokens([
      line_no_tokens + matching_tokens
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(line_no_tokens) == ['line_no'] * len(line_no_tokens)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)

  def test_should_annotate_shorter_target_annotation_in_longer_line(self):
    pre_tokens = _tokens_for_text('pre')
    matching_tokens = _tokens_for_text('this is matching')
    post_tokens = _tokens_for_text('post')
    target_annotations = [
      TargetAnnotation('this is matching', TAG1)
    ]
    doc = _document_for_tokens([
      pre_tokens + matching_tokens + post_tokens
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(pre_tokens) == [None] * len(pre_tokens)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(post_tokens) == [None] * len(post_tokens)

  def test_should_annotate_shorter_target_annotation_fuzzily(self):
    pre_tokens = _tokens_for_text('pre')
    matching_tokens = _tokens_for_text('this is matching')
    post_tokens = _tokens_for_text('post')
    target_annotations = [
      TargetAnnotation('this is. matching', TAG1)
    ]
    doc = _document_for_tokens([
      pre_tokens + matching_tokens + post_tokens
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(pre_tokens) == [None] * len(pre_tokens)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(post_tokens) == [None] * len(post_tokens)

  def test_should_annotate_multiple_shorter_target_annotation_in_longer_line(self):
    pre_tokens = _tokens_for_text('pre')
    matching_tokens_tag_1 = _tokens_for_text('this is matching')
    mid_tokens = _tokens_for_text('mid')
    matching_tokens_tag_2 = _tokens_for_text('also good')
    post_tokens = _tokens_for_text('post')
    target_annotations = [
      TargetAnnotation('this is matching', TAG1),
      TargetAnnotation('also good', TAG2)
    ]
    doc = _document_for_tokens([
      pre_tokens + matching_tokens_tag_1 + mid_tokens + matching_tokens_tag_2 + post_tokens
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(pre_tokens) == [None] * len(pre_tokens)
    assert _get_tags_of_tokens(matching_tokens_tag_1) == [TAG1] * len(matching_tokens_tag_1)
    assert _get_tags_of_tokens(mid_tokens) == [None] * len(mid_tokens)
    assert _get_tags_of_tokens(matching_tokens_tag_2) == [TAG2] * len(matching_tokens_tag_2)
    assert _get_tags_of_tokens(post_tokens) == [None] * len(post_tokens)

  def test_should_not_annotate_shorter_target_annotation_in_longer_line_multiple_times(self):
    pre_tokens = _tokens_for_text('pre')
    matching_tokens = _tokens_for_text('this is matching')
    post_tokens = _tokens_for_text('post')
    first_line_tokens = pre_tokens + matching_tokens + post_tokens
    similar_line_tokens = _copy_tokens(first_line_tokens)
    target_annotations = [
      TargetAnnotation('this is matching', TAG1)
    ]
    doc = _document_for_tokens([
      first_line_tokens,
      similar_line_tokens
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(similar_line_tokens) == [None] * len(similar_line_tokens)

  def test_should_annotate_shorter_target_annotation_in_longer_line_multiple_times_if_enabled(self):
    pre_tokens = _tokens_for_text('pre')
    matching_tokens = _tokens_for_text('this is matching')
    post_tokens = _tokens_for_text('post')
    same_matching_tokens = _copy_tokens(matching_tokens)
    target_annotations = [
      TargetAnnotation('this is matching', TAG1, match_multiple=True)
    ]
    doc = _document_for_tokens([
      pre_tokens + matching_tokens + post_tokens,
      _copy_tokens(pre_tokens) + same_matching_tokens + _copy_tokens(post_tokens)
    ])
    MatchingAnnotator(target_annotations).annotate(doc)
    assert _get_tags_of_tokens(matching_tokens) == [TAG1] * len(matching_tokens)
    assert _get_tags_of_tokens(same_matching_tokens) == [TAG1] * len(same_matching_tokens)

class TestMatchingAnnotatorSubAnnotations(object):
  def test_should_annotate_sub_tag_exactly_matching_without_begin_prefix(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is matching', TAG2, sub_annotations=[
        TargetAnnotation('this', TAG1)
      ])
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=False).annotate(doc)
    assert [doc.get_sub_tag(x) for x in matching_tokens] == [TAG1, None, None]

  def test_should_annotate_sub_tag_exactly_matching_with_begin_prefix(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is matching', TAG2, sub_annotations=[
        TargetAnnotation('this', TAG1)
      ])
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=True).annotate(doc)
    assert [doc.get_sub_tag(x) for x in matching_tokens] == [B_TAG_1, None, None]

  def test_should_annotate_sub_tag_case_insensitive(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is matching', TAG2, sub_annotations=[
        TargetAnnotation('This', TAG1)
      ])
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=False).annotate(doc)
    assert [doc.get_sub_tag(x) for x in matching_tokens] == [TAG1, None, None]

  def test_should_annotate_multiple_sub_tag_exactly_matching(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is matching', TAG2, sub_annotations=[
        TargetAnnotation('this', TAG1),
        TargetAnnotation('is', TAG2)
      ])
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=False).annotate(doc)
    assert [doc.get_sub_tag(x) for x in matching_tokens] == [TAG1, TAG2, None]

  def test_should_annotate_multiple_sub_annotations_with_same_sub_tag(self):
    matching_tokens = _tokens_for_text('this is matching')
    target_annotations = [
      TargetAnnotation('this is matching', TAG2, sub_annotations=[
        TargetAnnotation('this', TAG1),
        TargetAnnotation('is', TAG1)
      ])
    ]
    doc = _document_for_tokens([matching_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=False).annotate(doc)
    assert [doc.get_sub_tag(x) for x in matching_tokens] == [TAG1, TAG1, None]

  def test_should_annotate_sub_tag_across_multiple_tokens(self):
    sub_matching_tokens = _tokens_for_text('this is matching')
    tag_matching_tokens = (
      _tokens_for_text('something before') +
      sub_matching_tokens +
      _tokens_for_text('more text to come')
    )
    all_tokens = (
      _tokens_for_text('not matching') + tag_matching_tokens + _tokens_for_text('and there')
    )
    target_annotations = [
      TargetAnnotation(_tokens_to_text(tag_matching_tokens), TAG2, sub_annotations=[
        TargetAnnotation(_tokens_to_text(sub_matching_tokens), TAG1)
      ])
    ]
    doc = _document_for_tokens([all_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=False).annotate(doc)
    assert [doc.get_sub_tag(x) for x in sub_matching_tokens] == [TAG1, TAG1, TAG1]

  def test_should_annotate_sub_tag_across_multiple_tokens_with_junk_characters(self):
    junk_char = ','
    sub_matching_tokens = _tokens_for_text('this is matching' + junk_char)
    tag_matching_tokens = (
      _tokens_for_text('something before') +
      sub_matching_tokens +
      _tokens_for_text('more text to come')
    )
    all_tokens = (
      _tokens_for_text('not matching') + tag_matching_tokens + _tokens_for_text('and there')
    )
    tag_matching_text = _tokens_to_text(tag_matching_tokens).replace(junk_char, '')
    sub_matching_text = _tokens_to_text(sub_matching_tokens).replace(junk_char, '')
    target_annotations = [
      TargetAnnotation(tag_matching_text, TAG2, sub_annotations=[
        TargetAnnotation(sub_matching_text, TAG1)
      ])
    ]
    doc = _document_for_tokens([all_tokens])
    MatchingAnnotator(target_annotations, use_tag_begin_prefix=False).annotate(doc)
    assert [doc.get_sub_tag(x) for x in sub_matching_tokens] == [TAG1, TAG1, TAG1]
