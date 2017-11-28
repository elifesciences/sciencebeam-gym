from sciencebeam_gym.alignment.WordSequenceMatcher import (
  WordSequenceMatcher
)

WORD_1 = 'word1'
WORD_2 = 'word2'
WORD_3 = 'word3'

class TestWordSequenceMatcher(object):
  def test_should_not_match_different_words(self):
    sm = WordSequenceMatcher(None, WORD_1, WORD_2)
    matching_blocks = sm.get_matching_blocks()
    assert len(matching_blocks) == 0

  def test_should_match_same_words_standalone(self):
    sm = WordSequenceMatcher(None, WORD_1, WORD_1)
    matching_blocks = sm.get_matching_blocks()
    assert matching_blocks == [(
      0,
      0,
      len(WORD_1)
    )]

  def test_should_match_same_words_within_other_words(self):
    a_words = ['pre_a__', WORD_1, 'post_a']
    b_words = ['pre_b', WORD_1, 'post_b']
    sm = WordSequenceMatcher(
      None,
      ' '.join(a_words),
      ' '.join(b_words)
    )
    matching_blocks = sm.get_matching_blocks()
    assert matching_blocks == [(
      len(a_words[0]) + 1,
      len(b_words[0]) + 1,
      len(WORD_1)
    )]

  def test_should_match_same_words_standalone_ignore_comma_after_word(self):
    sm = WordSequenceMatcher(None, WORD_1 + ',', WORD_1)
    matching_blocks = sm.get_matching_blocks()
    assert matching_blocks == [(
      0,
      0,
      len(WORD_1)
    )]
