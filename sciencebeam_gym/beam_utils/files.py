from itertools import islice

import apache_beam as beam

from sciencebeam_gym.beam_utils.csv import (
  ReadDictCsv
)

from sciencebeam_gym.beam_utils.io import (
  find_matching_filenames
)

from sciencebeam_gym.beam_utils.utils import (
  GroupTransforms
)

from sciencebeam_gym.utils.file_list import (
  load_file_list
)

def find_matching_filenames_with_limit(pattern, limit=None):
  return islice(
    find_matching_filenames(pattern),
    limit
  )

def ReadFileList(file_list_path, column, limit=None):
  file_list = load_file_list(file_list_path, column=column, limit=limit)
  return beam.Create(file_list)

def DeferredReadFileList(file_list_path, column, limit=None):
  return GroupTransforms(lambda p: (
    p |
    "ReadFileUrls" >> ReadDictCsv(file_list_path, limit=limit) |
    "TranslateFileUrls" >> beam.Map(lambda row: row[column])
  ))

def FindFiles(file_pattern, limit=None):
  file_list = list(find_matching_filenames_with_limit(file_pattern, limit=limit))
  return beam.Create(file_list)

def DeferredFindFiles(file_pattern, limit=None):
  return GroupTransforms(lambda p: (
    p |
    beam.Create([file_pattern]) |
    "FindFiles" >> beam.FlatMap(
      lambda pattern: find_matching_filenames_with_limit(pattern, limit)
    )
  ))
