from __future__ import division

import logging

from sciencebeam_gym.preprocess.annotation.fuzzy_match import (
    remove_junk,
    invert_index_ranges,
    FuzzyMatchResult,
    fuzzy_match,
    DOT_IS_JUNK
)


def setup_module():
    logging.basicConfig(level='DEBUG')


class TestRemoveJunk(object):
    def test_should_keep_str_without_junk(self):
        assert remove_junk('abc', DOT_IS_JUNK) == 'abc'

    def test_should_remove_dots_after_capitals(self):
        assert remove_junk('P.O. Box', DOT_IS_JUNK) == 'PO Box'

    def test_should_remove_asterisk_after_capitals(self):
        assert remove_junk('Mr Beam*') == 'Mr Beam'

    def test_should_remove_repeating_characters(self):
        assert remove_junk('Mr Beeeam') == 'Mr Beam'


class TestInvertIndexRanges(object):
    def test_should_return_empty_for_empty_range(self):
        assert list(invert_index_ranges([], 0, 0)) == list([])

    def test_should_return_whole_range_for_empty_range_list(self):
        assert list(invert_index_ranges([], 0, 10)) == list([(0, 10)])

    def test_should_exclude_range_in_the_beginning(self):
        assert list(invert_index_ranges([(0, 3)], 0, 10)) == list([(3, 10)])

    def test_should_exclude_range_in_the_beginning_beyond_start(self):
        assert list(invert_index_ranges([(0, 13)], 10, 20)) == list([(13, 20)])

    def test_should_exclude_range_in_the_middle(self):
        assert list(invert_index_ranges([(4, 7)], 0, 10)) == list([(0, 4), (7, 10)])

    def test_should_exclude_range_at_the_end(self):
        assert list(invert_index_ranges([(7, 10)], 0, 10)) == list([(0, 7)])

    def test_should_exclude_range_at_the_end_beyond_end(self):
        assert list(invert_index_ranges([(7, 100)], 0, 10)) == list([(0, 7)])


class TestFuzzyMatch(object):
    def test_match_count_should_be_the_same_independent_of_order(self):
        s1 = 'this is a some sequence'
        choice = 'this is another sequence'
        fm_1 = fuzzy_match(s1, choice)
        fm_2 = fuzzy_match(choice, s1)
        assert fm_1.match_count() == fm_2.match_count()


class TestFuzzyMatchResult(object):
    def test_exact_match(self):
        fm = FuzzyMatchResult('abc', 'abc', [(0, 0, 3)])
        assert fm.has_match()
        assert fm.match_count() == 3
        assert fm.ratio() == 1.0
        assert fm.a_ratio() == 1.0
        assert fm.b_ratio() == 1.0
        assert fm.b_gap_ratio() == 1.0
        assert fm.a_index_range() == (0, 3)
        assert fm.b_index_range() == (0, 3)

    def test_no_match(self):
        fm = FuzzyMatchResult('abc', 'xyz', [])
        assert not fm.has_match()
        assert fm.match_count() == 0

    def test_partial_match(self):
        fm = FuzzyMatchResult('abx', 'aby', [(0, 0, 2)])
        assert fm.has_match()
        assert fm.match_count() == 2
        assert fm.ratio() == 1.0
        assert fm.a_ratio() == 2 / 3
        assert fm.b_ratio() == 2 / 3
        assert fm.b_gap_ratio() == 2 / 3
        assert fm.a_index_range() == (0, 2)
        assert fm.b_index_range() == (0, 2)

    def test_partial_match_ignore_junk_at_the_end_of_a(self):
        fm = FuzzyMatchResult('ab.', 'ab', [(0, 0, 2)], isjunk=lambda s, i: s[i] == '.')
        assert fm.has_match()
        assert fm.match_count() == 2
        assert fm.ratio() == 1.0
        assert fm.a_ratio() == 1.0
        assert fm.b_ratio() == 1.0
        assert fm.b_gap_ratio() == 1.0
        assert fm.a_index_range() == (0, 2)
        assert fm.b_index_range() == (0, 2)

    def test_partial_match_ignore_junk_at_the_end_of_b(self):
        fm = FuzzyMatchResult('ab', 'ab.', [(0, 0, 2)], isjunk=lambda s, i: s[i] == '.')
        assert fm.has_match()
        assert fm.match_count() == 2
        assert fm.ratio() == 1.0
        assert fm.a_ratio() == 1.0
        assert fm.b_ratio() == 1.0
        assert fm.b_gap_ratio() == 1.0
        assert fm.a_index_range() == (0, 2)
        assert fm.b_index_range() == (0, 2)

    def test_partial_match_ignore_junk_in_the_middle_of_a(self):
        fm = FuzzyMatchResult('a.b', 'ab', [(0, 0, 1), (2, 1, 1)], isjunk=lambda s, i: s[i] == '.')
        assert fm.has_match()
        assert fm.match_count() == 2
        assert fm.ratio() == 1.0
        assert fm.a_ratio() == 1.0
        assert fm.b_ratio() == 1.0
        assert fm.b_gap_ratio() == 1.0
        assert fm.a_index_range() == (0, 3)
        assert fm.b_index_range() == (0, 2)

    def test_partial_match_ignore_junk_in_the_middle_of_b(self):
        fm = FuzzyMatchResult('ab', 'a.b', [(0, 0, 1), (1, 2, 1)], isjunk=lambda s, i: s[i] == '.')
        assert fm.has_match()
        assert fm.match_count() == 2
        assert fm.ratio() == 1.0
        assert fm.a_ratio() == 1.0
        assert fm.b_ratio() == 1.0
        assert fm.b_gap_ratio() == 1.0
        assert fm.a_index_range() == (0, 2)
        assert fm.b_index_range() == (0, 3)

    def test_should_not_double_count_matching_junk(self):
        fm = FuzzyMatchResult('a.b', 'a.b', [(0, 0, 3)], isjunk=lambda s, i: s[i] == '.')
        assert fm.has_match()
        assert fm.match_count() == 3
        assert fm.ratio() == 1.0
        assert fm.a_ratio() == 1.0
        assert fm.b_ratio() == 1.0
        assert fm.b_gap_ratio() == 1.0
        assert fm.a_index_range() == (0, 3)
        assert fm.b_index_range() == (0, 3)

    def test_a_split_no_match(self):
        fm = FuzzyMatchResult('abc', 'xyz', [])
        fm_1, fm_2 = fm.a_split_at(2)

        assert not fm_1.has_match()
        assert fm_1.a == 'ab'
        assert fm_1.b == 'xyz'

        assert not fm_2.has_match()
        assert fm_2.a == 'c'
        assert fm_2.b == 'xyz'

    def test_b_split_no_match(self):
        fm = FuzzyMatchResult('abc', 'xyz', [])
        fm_1, fm_2 = fm.b_split_at(2)

        assert not fm_1.has_match()
        assert fm_1.a == 'abc'
        assert fm_1.b == 'xy'

        assert not fm_2.has_match()
        assert fm_2.a == 'abc'
        assert fm_2.b == 'z'

    def test_a_split_exact_match(self):
        fm = FuzzyMatchResult('abc', 'abc', [(0, 0, 3)])
        fm_1, fm_2 = fm.a_split_at(2)

        assert fm_1.a == 'ab'
        assert fm_1.b == 'abc'
        assert fm_1.has_match()
        assert fm_1.ratio() == 1.0
        assert fm_1.a_ratio() == 1.0
        assert fm_1.b_ratio() == 2 / 3
        assert fm_1.b_gap_ratio() == 2 / 3
        assert fm_1.a_index_range() == (0, 2)
        assert fm_1.b_index_range() == (0, 2)

        assert fm_2.a == 'c'
        assert fm_2.b == 'abc'
        assert fm_2.has_match()
        assert fm_2.ratio() == 1.0
        assert fm_2.a_ratio() == 1.0
        assert fm_2.b_ratio() == 1 / 3
        assert fm_2.b_gap_ratio() == 1 / 3
        assert fm_2.a_index_range() == (0, 1)
        assert fm_2.b_index_range() == (0, 1)
