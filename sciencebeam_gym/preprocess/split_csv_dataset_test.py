from collections import namedtuple

from sciencebeam_gym.preprocess.split_csv_dataset import (
  extract_proportions_from_args,
  split_rows,
  output_filenames_for_names
)

def create_args(**kwargs):
  return namedtuple('args', kwargs.keys())(**kwargs)

class TestExtractProportionsFromArgs(object):
  def test_should_create_train_test_split_with_only_train_specified(self):
    assert extract_proportions_from_args(
      create_args(train=0.6, test=None, validation=None)
    ) == [('train', 0.6), ('test', 0.4)]

  def test_should_create_train_test_validation_split_with_train_and_test_specified(self):
    assert extract_proportions_from_args(
      create_args(train=0.6, test=0.3, validation=None)
    ) == [('train', 0.6), ('test', 0.3), ('validation', 0.1)]

  def test_should_not_add_validation_if_remaining_percentage_is_zero(self):
    assert extract_proportions_from_args(
      create_args(train=0.6, test=0.4, validation=None)
    ) == [('train', 0.6), ('test', 0.4)]

class TestSplitRows(object):
  def test_should_split_train_test(self):
    assert split_rows(list(range(10)), [0.6, 0.4]) == [
      list(range(6)),
      list(range(6, 10))
    ]

  def test_should_split_train_test_validation(self):
    assert split_rows(list(range(10)), [0.6, 0.3, 0.1]) == [
      list(range(6)),
      list(range(6, 9)),
      list(range(9, 10))
    ]

  def test_should_round_down(self):
    assert split_rows(list(range(11)), [0.6, 0.4]) == [
      list(range(6)),
      list(range(6, 10))
    ]

  def test_should_fill_last_chunk_if_enabled(self):
    assert split_rows(list(range(11)), [0.6, 0.4], fill=True) == [
      list(range(6)),
      list(range(6, 11))
    ]

class TestGetOutputFilenamesForNames(object):
  def test_should_add_name_and_ext_with_path_sep_if_out_ends_with_slash(self):
    assert output_filenames_for_names(
      ['train', 'test'], 'out/', '.tsv'
    ) == ['out/train.tsv', 'out/test.tsv']

  def test_should_add_name_and_ext_with_hyphen_if_out_does_not_end_with_slash(self):
    assert output_filenames_for_names(
      ['train', 'test'], 'out', '.tsv'
    ) == ['out-train.tsv', 'out-test.tsv']
