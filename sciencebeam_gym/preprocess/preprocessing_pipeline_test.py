from contextlib import contextmanager
import logging
from mock import Mock, patch, DEFAULT

import pytest

import apache_beam as beam

from sciencebeam_utils.beam_utils.utils import (
  TransformAndLog
)

from sciencebeam_utils.beam_utils.testing import (
  BeamTest,
  TestPipeline,
  get_current_test_context,
  get_counter_value
)

from sciencebeam_utils.utils.collection import (
  extend_dict
)

from sciencebeam_gym.preprocess.preprocessing_pipeline import (
  parse_args,
  configure_pipeline,
  MetricCounters
)

PREPROCESSING_PIPELINE = 'sciencebeam_gym.preprocess.preprocessing_pipeline'

BASE_DATA_PATH = 'base'
PDF_PATH = '*/*.pdf'
XML_PATH = '*/*.xml'

PDF_FILE_1 = '1/file.pdf'
XML_FILE_1 = '1/file.xml'
PDF_FILE_2 = '2/file.pdf'
XML_FILE_2 = '2/file.xml'
PDF_XML_FILE_LIST_FILE_1 = 'pdf-xml-files.tsv'

def get_logger():
  return logging.getLogger(__name__)

def fake_content(path):
  return 'fake content: %s' % path

def fake_lxml_for_pdf(pdf, path, page_range=None):
  return 'fake lxml for pdf: %s (%s) [%s]' % (pdf, path, page_range)

fake_svg_page = lambda i=0: 'fake svg page: %d' % i
fake_pdf_png_page = lambda i=0: 'fake pdf png page: %d' % i
fake_block_png_page = lambda i=0: 'fake block png page: %d' % i

def get_global_tfrecords_mock():
  # workaround for mock that would get serialized/deserialized before being invoked
  return get_current_test_context().tfrecords_mock

@contextmanager
def patch_preprocessing_pipeline(**kwargs):
  always_mock = {
    'find_file_pairs_grouped_by_parent_directory_or_name',
    'read_all_from_path',
    'pdf_bytes_to_png_pages',
    'convert_pdf_bytes_to_lxml',
    'convert_and_annotate_lxml_content',
    'svg_page_to_blockified_png_bytes',
    'save_svg_roots',
    'save_pages',
    'evaluate_document_by_page',
    'ReadDictCsv'
  }
  tfrecords_mock = Mock(name='tfrecords_mock')

  def DummyWritePropsToTFRecord(file_path, extract_props):
    return TransformAndLog(beam.Map(
      lambda v: tfrecords_mock(file_path, list(extract_props(v)))
    ), log_fn=lambda x: get_logger().info('tfrecords: %s', x))

  with patch.multiple(
    PREPROCESSING_PIPELINE,
    WritePropsToTFRecord=DummyWritePropsToTFRecord,
    **{
      k: kwargs.get(k, DEFAULT)
      for k in always_mock
    }
  ) as mocks:
    get_current_test_context().mocks = mocks
    mocks['read_all_from_path'].side_effect = fake_content
    mocks['convert_pdf_bytes_to_lxml'].side_effect = fake_lxml_for_pdf
    yield extend_dict(
      mocks,
      {'tfrecords': tfrecords_mock}
    )

MIN_ARGV = [
  '--data-path=' + BASE_DATA_PATH,
  '--pdf-path=' + PDF_PATH,
  '--xml-path=' + XML_PATH,
  '--save-svg'
]

def get_default_args():
  return parse_args([
    '--data-path=' + BASE_DATA_PATH,
    '--pdf-path=' + PDF_PATH,
    '--xml-path=' + XML_PATH,
    '--save-svg'
  ])

def page_uri_suffix(page_no):
  return '#page%d' % page_no

def _expected_tfrecord_props(pdf_file, page_no=1):
  return {
    'input_uri': pdf_file + page_uri_suffix(page_no),
    'annotation_uri': pdf_file + '.annot' + page_uri_suffix(page_no),
    'input_image': fake_pdf_png_page(page_no),
    'annotation_image': fake_block_png_page(page_no),
    'page_no': page_no
  }

def _setup_mocks_for_pages(mocks, page_no_list, file_count=1):
  mocks['convert_and_annotate_lxml_content'].return_value = [
    fake_svg_page(i) for i in page_no_list
  ]
  mocks['pdf_bytes_to_png_pages'].return_value = [
    fake_pdf_png_page(i) for i in page_no_list
  ]
  mocks['svg_page_to_blockified_png_bytes'].side_effect = [
    fake_block_png_page(i)
    for _ in range(file_count)
    for i in page_no_list
  ]

@pytest.mark.slow
class TestConfigurePipeline(BeamTest):
  def test_should_pass_pdf_and_xml_patterns_to_find_file_pairs_grouped_by_parent_directory(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.base_data_path = 'base'
      opt.pdf_path = 'pdf'
      opt.xml_path = 'xml'
      with TestPipeline() as p:
        mocks['find_file_pairs_grouped_by_parent_directory_or_name'].return_value = []
        configure_pipeline(p, opt)

      mocks['find_file_pairs_grouped_by_parent_directory_or_name'].assert_called_with(
        ['base/pdf', 'base/xml']
      )

  def test_should_pass_lxml_and_xml_patterns_to_find_file_pairs_grouped_by_parent_directory(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.base_data_path = 'base'
      opt.pdf_path = ''
      opt.lxml_path = 'lxml'
      opt.xml_path = 'xml'
      with TestPipeline() as p:
        mocks['find_file_pairs_grouped_by_parent_directory_or_name'].return_value = []
        configure_pipeline(p, opt)

      mocks['find_file_pairs_grouped_by_parent_directory_or_name'].assert_called_with(
        ['base/lxml', 'base/xml']
      )

  def test_should_write_tfrecords_from_pdf_xml_file_list(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.pdf_path = None
      opt.xml_path = None
      opt.pdf_xml_file_list = '.temp/file-list.tsv'
      opt.save_tfrecords = True
      with TestPipeline() as p:
        mocks['ReadDictCsv'].return_value = beam.Create([{
          'source_url': PDF_FILE_1,
          'xml_url': XML_FILE_1
        }])
        _setup_mocks_for_pages(mocks, [1])
        configure_pipeline(p, opt)

      mocks['ReadDictCsv'].assert_called_with(opt.pdf_xml_file_list, limit=None)
      mocks['tfrecords'].assert_called_with(opt.output_path + '/data', [
        _expected_tfrecord_props(PDF_FILE_1)
      ])

  def test_should_write_multiple_tfrecords_from_pdf_xml_file_list(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.pdf_path = None
      opt.xml_path = None
      opt.pdf_xml_file_list = '.temp/file-list.tsv'
      opt.save_tfrecords = True
      with TestPipeline() as p:
        mocks['ReadDictCsv'].return_value = beam.Create([{
          'source_url': PDF_FILE_1,
          'xml_url': XML_FILE_1
        }, {
          'source_url': PDF_FILE_2,
          'xml_url': XML_FILE_2
        }])
        _setup_mocks_for_pages(mocks, [1], file_count=2)
        configure_pipeline(p, opt)

      mocks['ReadDictCsv'].assert_called_with(opt.pdf_xml_file_list, limit=None)
      for pdf_file in [PDF_FILE_1, PDF_FILE_2]:
        mocks['tfrecords'].assert_any_call(opt.output_path + '/data', [
          _expected_tfrecord_props(pdf_file)
        ])
      assert mocks['tfrecords'].call_count == 2

  def test_should_pass_limit_to_read_dict_csv(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.pdf_path = None
      opt.xml_path = None
      opt.pdf_xml_file_list = '.temp/file-list.tsv'
      opt.limit = 1
      opt.save_tfrecords = True
      with TestPipeline() as p:
        mocks['ReadDictCsv'].return_value = beam.Create([{
          'source_url': PDF_FILE_1,
          'xml_url': XML_FILE_1
        }])
        _setup_mocks_for_pages(mocks, [1])
        configure_pipeline(p, opt)

      mocks['ReadDictCsv'].assert_called_with(opt.pdf_xml_file_list, limit=opt.limit)
      assert mocks['tfrecords'].call_count == 1

  def test_should_pass_limit_to_find_file_pairs_grouped_by_parent_directory_or_name(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.base_data_path = 'base'
      opt.pdf_path = 'pdf'
      opt.lxml_path = ''
      opt.xml_path = 'xml'
      opt.save_tfrecords = True
      opt.limit = 1
      with TestPipeline() as p:
        mocks['find_file_pairs_grouped_by_parent_directory_or_name'].return_value = [
          (PDF_FILE_1, XML_FILE_1),
          (PDF_FILE_2, XML_FILE_2)
        ]
        configure_pipeline(p, opt)

      assert mocks['tfrecords'].call_count == 1

  def test_should_write_tfrecords_from_pdf_xml_path(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.save_tfrecords = True
      with TestPipeline() as p:
        mocks['find_file_pairs_grouped_by_parent_directory_or_name'].return_value = [
          (PDF_FILE_1, XML_FILE_1)
        ]
        _setup_mocks_for_pages(mocks, [1])
        configure_pipeline(p, opt)

      mocks['tfrecords'].assert_called_with(opt.output_path + '/data', [
        _expected_tfrecord_props(PDF_FILE_1)
      ])

  def test_should_write_multiple_tfrecords_and_count_pages(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.save_tfrecords = True
      with TestPipeline() as p:
        mocks['find_file_pairs_grouped_by_parent_directory_or_name'].return_value = [
          (PDF_FILE_1, XML_FILE_1)
        ]
        _setup_mocks_for_pages(mocks, [1, 2])
        configure_pipeline(p, opt)

        p_result = p.run()
        assert get_counter_value(p_result, MetricCounters.FILE_PAIR) == 1
        assert get_counter_value(p_result, MetricCounters.PAGE) == 2
        assert get_counter_value(p_result, MetricCounters.FILTERED_PAGE) is None

      mocks['tfrecords'].assert_called_with(opt.output_path + '/data', [
        _expected_tfrecord_props(PDF_FILE_1, page_no=i)
        for i in [1, 2]
      ])

  def test_should_not_write_tfrecord_below_annotation_threshold_and_count_pages(self):
    custom_mocks = dict(
      evaluate_document_by_page=lambda _: [{
        'percentage': {
          # low percentage of None (no annotation, include)
          None: 0.1
        }
      }, {
        'percentage': {
          # low percentage of None (no annotation, exclude)
          None: 0.9
        }
      }]
    )
    with patch_preprocessing_pipeline(**custom_mocks) as mocks:
      opt = get_default_args()
      opt.save_tfrecords = True
      opt.min_annotation_percentage = 0.5
      with TestPipeline() as p:
        mocks['find_file_pairs_grouped_by_parent_directory_or_name'].return_value = [
          (PDF_FILE_1, XML_FILE_1)
        ]
        _setup_mocks_for_pages(mocks, [1, 2])
        configure_pipeline(p, opt)

        p_result = p.run()
        assert get_counter_value(p_result, MetricCounters.FILE_PAIR) == 1
        assert get_counter_value(p_result, MetricCounters.PAGE) == 2
        assert get_counter_value(p_result, MetricCounters.FILTERED_PAGE) == 1

      mocks['tfrecords'].assert_called_with(opt.output_path + '/data', [
        _expected_tfrecord_props(PDF_FILE_1, page_no=i)
        for i in [1]
      ])

  def test_should_only_process_selected_pages(self):
    with patch_preprocessing_pipeline() as mocks:
      opt = get_default_args()
      opt.save_tfrecords = True
      opt.save_png = True
      opt.pages = (1, 3)
      with TestPipeline() as p:
        mocks['find_file_pairs_grouped_by_parent_directory_or_name'].return_value = [
          (PDF_FILE_1, XML_FILE_1)
        ]
        _setup_mocks_for_pages(mocks, [1, 2])
        configure_pipeline(p, opt)

      assert mocks['convert_pdf_bytes_to_lxml'].called
      assert mocks['convert_pdf_bytes_to_lxml'].call_args[1].get('page_range') == opt.pages

      assert mocks['pdf_bytes_to_png_pages'].called
      assert mocks['pdf_bytes_to_png_pages'].call_args[1].get('page_range') == opt.pages

class TestParseArgs(object):
  def test_should_raise_error_without_arguments(self):
    with pytest.raises(SystemExit):
      parse_args([])

  def test_should_not_raise_error_with_minimum_arguments(self):
    parse_args(['--data-path=test', '--pdf-path=test', '--xml-path=test', '--save-svg'])

  def test_should_not_raise_error_with_lxml_path_instead_of_pdf_path(self):
    parse_args(['--data-path=test', '--lxml-path=test', '--xml-path=test', '--save-svg'])

  def test_should_raise_error_if_no_output_option_specified(self):
    with pytest.raises(SystemExit):
      parse_args(['--data-path=test', '--pdf-path=test', '--xml-path=test'])

  def test_should_raise_error_if_pdf_and_lxml_path_are_specified(self):
    with pytest.raises(SystemExit):
      parse_args([
        '--data-path=test', '--pdf-path=test', '--lxml-path=test', '--xml-path=test',
        '--save-svg'
      ])

  def test_should_raise_error_if_pdf_path_specified_without_xml_path(self):
    with pytest.raises(SystemExit):
      parse_args(['--data-path=test', '--pdf-path=test', '--save-svg'])

  def test_should_not_raise_error_if_pdf_xml_file_list_specified_without_xml_path(self):
    parse_args(['--data-path=test', '--pdf-xml-file-list=test', '--save-svg'])

  def test_should_not_raise_error_with_save_lxml_path_together_with_pdf_path(self):
    parse_args(['--data-path=test', '--pdf-path=test', '--save-lxml', '--xml-path=test'])

  def test_should_not_raise_error_with_save_lxml_path_together_with_pdf_xml_file_list(self):
    parse_args(['--data-path=test', '--pdf-xml-file-list=test', '--save-lxml', '--xml-path=test'])

  def test_should_raise_error_if_save_lxml_specified_without_pdf_path(self):
    with pytest.raises(SystemExit):
      parse_args(['--data-path=test', '--lxml-path=test', '--save-lxml', '--xml-path=test'])

  def test_should_raise_error_if_save_png_is_specified_without_pdf_path(self):
    with pytest.raises(SystemExit):
      parse_args(['--data-path=test', '--lxml-path=test', '--save-png', '--xml-path=test'])

  def test_should_not_raise_error_with_save_png_path_together_with_pdf_path(self):
    parse_args(['--data-path=test', '--pdf-path=test', '--save-png', '--xml-path=test'])

  def test_should_not_raise_error_with_save_png_path_together_with_pdf_xml_file_list(self):
    parse_args(['--data-path=test', '--pdf-xml-file-list=test', '--save-png', '--xml-path=test'])

  def test_should_raise_error_if_image_width_was_specified_without_image_height(self):
    with pytest.raises(SystemExit):
      parse_args([
        '--data-path=test', '--pdf-path=test', '--xml-path=test',
        '--save-png', '--image-width=100'
      ])

  def test_should_raise_error_if_image_height_was_specified_without_image_width(self):
    with pytest.raises(SystemExit):
      parse_args([
        '--data-path=test', '--pdf-path=test', '--xml-path=test',
        '--save-png', '--image-height=100'
      ])

  def test_should_not_raise_error_if_both_image_width_and_height_are_specified(self):
    parse_args([
      '--data-path=test', '--pdf-path=test', '--xml-path=test',
      '--save-png', '--image-width=100', '--image-height=100'
    ])

  def test_should_raise_error_if_save_tfrecords_specified_without_pdf_path(self):
    with pytest.raises(SystemExit):
      parse_args(['--data-path=test', '--lxml-path=test', '--xml-path=test', '--save-tfrecords'])

  def test_should_not_raise_error_if_save_tfrecords_specified_with_pdf_path(self):
    parse_args(['--data-path=test', '--pdf-path=test', '--xml-path=test', '--save-tfrecords'])

  def test_should_not_raise_error_if_save_tfrecords_specified_with_pdf_xml_file_list(self):
    parse_args([
      '--data-path=test', '--pdf-xml-file-list=test', '--xml-path=test', '--save-tfrecords'
    ])

  def test_should_have_none_page_range_by_default(self):
    assert parse_args(MIN_ARGV).pages is None

  def test_should_parse_pages_as_list(self):
    assert parse_args(MIN_ARGV + ['--pages=1-3']).pages == (1, 3)
