from mock import patch, MagicMock, DEFAULT

import pytest

from lxml import etree

from sciencebeam_gym.structured_document.svg import (
  SVG_DOC
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
  svg_page_to_blockified_png_bytes,
  group_file_pairs_by_parent_directory_or_name,
  convert_pdf_bytes_to_lxml,
  join_if_relative_path,
  change_ext,
  base_path_for_file_list,
  get_or_validate_base_path,
  get_output_file,
  parse_page_range,
)

PROCESSING_UTILS = 'sciencebeam_gym.preprocess.preprocessing_utils'

PDF_CONTENT_1 = b'pdf content 1'

class TestSvgPageToBlockifiedPngBytes(object):
  def test_should_parse_viewbox_and_pass_width_and_height_to_annotated_blocks_to_image(self):
    with patch.multiple(PROCESSING_UTILS, annotated_blocks_to_image=DEFAULT) as mocks:
      svg_page = etree.Element(SVG_DOC, attrib={
        'viewBox': '0 0 100.1 200.9'
      })
      color_map = {}
      image_size = (100, 200)
      svg_page_to_blockified_png_bytes(svg_page, color_map, image_size)
      call_args = mocks['annotated_blocks_to_image'].call_args
      kwargs = call_args[1]
      assert (kwargs.get('width'), kwargs.get('height')) == (100.1, 200.9)

class TestGroupFilePairsByParentDirectoryOrName(object):
  def test_should_return_empty_list_with_empty_input_file_lists(self):
    assert list(group_file_pairs_by_parent_directory_or_name([
      [],
      []
    ])) == []

  def test_should_group_single_file(self):
    assert list(group_file_pairs_by_parent_directory_or_name([
      ['parent1/file.x'],
      ['parent1/file.y']
    ])) == [('parent1/file.x', 'parent1/file.y')]

  def test_should_group_single_file_in_directory_with_different_names(self):
    assert list(group_file_pairs_by_parent_directory_or_name([
      ['parent1/file1.x'],
      ['parent1/file2.y']
    ])) == [('parent1/file1.x', 'parent1/file2.y')]

  def test_should_ignore_files_in_different_directories(self):
    assert list(group_file_pairs_by_parent_directory_or_name([
      ['parent1/file.x'],
      ['parent2/file.y']
    ])) == []

  def test_should_group_multiple_files_in_separate_parent_directories(self):
    assert list(group_file_pairs_by_parent_directory_or_name([
      ['parent1/file.x', 'parent2/file.x'],
      ['parent1/file.y', 'parent2/file.y']
    ])) == [
      ('parent1/file.x', 'parent1/file.y'),
      ('parent2/file.x', 'parent2/file.y')
    ]

  def test_should_group_multiple_files_in_same_parent_directory_with_same_name(self):
    assert list(group_file_pairs_by_parent_directory_or_name([
      ['parent1/file1.x', 'parent1/file2.x'],
      ['parent1/file1.y', 'parent1/file2.y']
    ])) == [
      ('parent1/file1.x', 'parent1/file1.y'),
      ('parent1/file2.x', 'parent1/file2.y')
    ]

  def test_should_group_multiple_files_in_same_parent_directory_with_same_name_gzipped(self):
    assert list(group_file_pairs_by_parent_directory_or_name([
      ['parent1/file1.x.gz', 'parent1/file2.x.gz'],
      ['parent1/file1.y.gz', 'parent1/file2.y.gz']
    ])) == [
      ('parent1/file1.x.gz', 'parent1/file1.y.gz'),
      ('parent1/file2.x.gz', 'parent1/file2.y.gz')
    ]

DEFAULT_PDF_TO_LXML_ARGS = ['-blocks', '-noImageInline', '-noImage', '-fullFontName']

LXML_CONTENT_1 = b'lxml content 1'

class TestConvertPdfBytesToLxml(object):
  def test_should_pass_pdf_content_and_default_args_to_process_input(self):
    mock = MagicMock()
    with patch.multiple(PROCESSING_UTILS, PdfToLxmlWrapper=mock):
      mock.return_value.process_input.return_value = LXML_CONTENT_1
      lxml_content = convert_pdf_bytes_to_lxml(PDF_CONTENT_1)
      mock.return_value.process_input.assert_called_with(
        PDF_CONTENT_1,
        DEFAULT_PDF_TO_LXML_ARGS
      )
      assert lxml_content == LXML_CONTENT_1

  def test_should_pass_include_page_range_in_args(self):
    mock = MagicMock()
    with patch.multiple(PROCESSING_UTILS, PdfToLxmlWrapper=mock):
      mock.return_value.process_input.return_value = LXML_CONTENT_1
      lxml_content = convert_pdf_bytes_to_lxml(PDF_CONTENT_1, page_range=(1, 3))
      mock.return_value.process_input.assert_called_with(
        PDF_CONTENT_1,
        DEFAULT_PDF_TO_LXML_ARGS + ['-f', '1', '-l', '3']
      )
      assert lxml_content == LXML_CONTENT_1

class TestJoinIfRelativePath(object):
  def test_should_return_path_if_base_path_is_none(self):
    assert join_if_relative_path(None, 'file') == 'file'

  def test_should_return_path_if_not_relative(self):
    assert join_if_relative_path('/parent', '/other/file') == '/other/file'

  def test_should_return_joined_path_if_relative(self):
    assert join_if_relative_path('/parent', 'file') == '/parent/file'

class TestChangeExt(object):
  def test_should_replace_simple_ext_with_simple_ext(self):
    assert change_ext('file.pdf', None, '.xml') == 'file.xml'

  def test_should_replace_simple_ext_with_combined_ext(self):
    assert change_ext('file.pdf', None, '.svg.zip') == 'file.svg.zip'

  def test_should_remove_gz_ext_before_replacing_ext(self):
    assert change_ext('file.pdf.gz', None, '.svg.zip') == 'file.svg.zip'

class TestBasePathForFileList(object):
  def test_should_return_empty_string_if_file_list_is_empty(self):
    assert base_path_for_file_list([]) == ''

  def test_should_return_empty_string_if_filename_is_empty(self):
    assert base_path_for_file_list(['']) == ''

  def test_should_return_parent_directory_of_single_file(self):
    assert base_path_for_file_list(['/base/path/1/file']) == '/base/path/1'

  def test_should_return_common_path_of_two_files(self):
    assert base_path_for_file_list(['/base/path/1/file', '/base/path/2/file']) == '/base/path'

  def test_should_return_common_path_of_two_files_using_protocol(self):
    assert base_path_for_file_list([
      'a://base/path/1/file', 'a://base/path/2/file'
    ]) == 'a://base/path'

  def test_should_return_common_path_of_two_files_using_forward_slash(self):
    assert base_path_for_file_list([
      '\\base\\path\\1\\file', '\\base\\path\\2\\file'
    ]) == '\\base\\path'

  def test_should_return_empty_string_if_no_common_path_was_found(self):
    assert base_path_for_file_list(['a://base/path/1/file', 'b://base/path/2/file']) == ''

  def test_should_return_common_path_ignoring_partial_name_match(self):
    assert base_path_for_file_list(['/base/path/file1', '/base/path/file2']) == '/base/path'

class TestGetOrValidateBasePath(object):
  def test_should_return_base_path_of_two_files_if_no_base_path_was_provided(self):
    assert get_or_validate_base_path(
      ['/base/path/1/file', '/base/path/2/file'],
      None
    ) == '/base/path'

  def test_should_return_passed_in_base_path_if_valid(self):
    assert get_or_validate_base_path(
      ['/base/path/1/file', '/base/path/2/file'],
      '/base'
    ) == '/base'

  def test_should_raise_error_if_passed_in_base_path_is_invalid(self):
    with pytest.raises(AssertionError):
      get_or_validate_base_path(
        ['/base/path/1/file', '/base/path/2/file'],
        '/base/other'
      )

class TestGetOutputFile(object):
  def test_should_return_output_file_with_path_and_change_ext(self):
    assert get_output_file(
      '/source/path/file.pdf',
      '/source',
      '/output',
      '.xml'
    ) == '/output/path/file.xml'

class TestPageRange(object):
  def test_should_parse_single_page_number_as_range(self):
    assert parse_page_range('1') == (1, 1)

  def test_should_parse_range_with_hyphen(self):
    assert parse_page_range('1-3') == (1, 3)

  def test_should_parse_range_with_spaces(self):
    assert parse_page_range(' 1 - 3 ') == (1, 3)

  def test_should_return_none_for_empty_range(self):
    assert parse_page_range('') is None
