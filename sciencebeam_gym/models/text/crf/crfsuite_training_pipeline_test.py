from mock import patch, Mock, ANY

import pytest

from sciencebeam_gym.utils.collection import (
  to_namedtuple
)

from sciencebeam_gym.structured_document import (
  SimpleStructuredDocument,
  SimpleLine,
  SimpleToken
)

from sciencebeam_gym.models.text.feature_extractor import (
  CV_TAG_SCOPE
)

import sciencebeam_gym.models.text.crf.crfsuite_training_pipeline as crfsuite_training_pipeline
from sciencebeam_gym.models.text.crf.crfsuite_training_pipeline import (
  load_and_convert_to_token_props,
  load_token_props_list_by_document,
  train_model,
  save_model,
  run,
  main
)

SOURCE_FILE_LIST_PATH = '.temp/source-file-list.lst'
CV_SOURCE_FILE_LIST_PATH = '.temp/cv-source-file-list.lst'

FILE_1 = 'file1.pdf'
FILE_2 = 'file2.pdf'
UNICODE_FILE_1 = u'file1\u1234.pdf'

MODEL_DATA = b'model data'

PAGE_RANGE = (2, 3)

TEXT_1 = 'text 1'

TAG_1 = 'tag1'
TAG_2 = 'tag2'

DEFAULT_ARGS = dict(
  source_file_column='url',
  cv_source_file_list=None,
  cv_source_file_column='url'
)

class TestLoadAndConvertToTokenProps(object):
  def test_should_load_and_convert_document(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_structured_document') as load_structured_document_mock:
      with patch.object(m, 'structured_document_to_token_props') as \
      structured_document_to_token_props_mock:

        load_and_convert_to_token_props(FILE_1, None, page_range=PAGE_RANGE)
        load_structured_document_mock.assert_called_with(
          FILE_1, page_range=PAGE_RANGE
        )
        structured_document_to_token_props_mock.assert_called_with(
          load_structured_document_mock.return_value
        )

  def test_should_load_and_convert_document_with_cv(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_structured_document') as load_structured_document_mock:
      with patch.object(m, 'structured_document_to_token_props') as \
      structured_document_to_token_props_mock:

        load_and_convert_to_token_props(FILE_1, FILE_2, page_range=PAGE_RANGE)
        load_structured_document_mock.assert_any_call(
          FILE_1, page_range=PAGE_RANGE
        )
        structured_document_to_token_props_mock.assert_called_with(
          load_structured_document_mock.return_value
        )

  def test_should_merge_doc_and_scope_cv_tag(self):
    structured_document = SimpleStructuredDocument(lines=[SimpleLine([
      SimpleToken(TEXT_1, tag=TAG_1)
    ])])
    cv_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
      SimpleToken(TEXT_1, tag=TAG_2)
    ])])
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_structured_document') as load_structured_document_mock:
      with patch.object(m, 'structured_document_to_token_props') as \
      structured_document_to_token_props_mock:

        load_structured_document_mock.side_effect = [
          structured_document,
          cv_structured_document
        ]
        load_and_convert_to_token_props(FILE_1, FILE_2, page_range=PAGE_RANGE)
        load_structured_document_mock.assert_any_call(
          FILE_1, page_range=PAGE_RANGE
        )
        structured_document_arg = structured_document_to_token_props_mock.call_args[0][0]
        assert [
          structured_document_arg.get_tag_by_scope(t)
          for t in structured_document_arg.iter_all_tokens()
        ] == [{None: TAG_1, CV_TAG_SCOPE: TAG_2}]

class TestLoadTokenPropsListByDocument(object):
  def test_should_load_single_file_without_cv(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_and_convert_to_token_props') as \
      load_and_convert_to_token_props_mock:

      result = load_token_props_list_by_document(
        [FILE_1], None, page_range=PAGE_RANGE, progress=False
      )
      load_and_convert_to_token_props_mock.assert_called_with(
        FILE_1, None, page_range=PAGE_RANGE
      )
      assert result == [load_and_convert_to_token_props_mock.return_value]

  def test_should_load_single_file_with_cv(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_and_convert_to_token_props') as \
      load_and_convert_to_token_props_mock:

      result = load_token_props_list_by_document(
        [FILE_1], [FILE_2], page_range=PAGE_RANGE, progress=False
      )
      load_and_convert_to_token_props_mock.assert_called_with(
        FILE_1, FILE_2, page_range=PAGE_RANGE
      )
      assert result == [load_and_convert_to_token_props_mock.return_value]

  def test_should_load_multiple_files(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_and_convert_to_token_props') as \
      load_and_convert_to_token_props_mock:

      return_values = [Mock(), Mock()]
      load_and_convert_to_token_props_mock.side_effect = return_values
      result = load_token_props_list_by_document(
        [FILE_1, FILE_2], None, page_range=PAGE_RANGE, progress=False
      )
      load_and_convert_to_token_props_mock.assert_any_call(
        FILE_1, None, page_range=PAGE_RANGE
      )
      load_and_convert_to_token_props_mock.assert_any_call(
        FILE_2, None, page_range=PAGE_RANGE
      )
      assert set(result) == set(return_values)

  def test_should_skip_files_with_errors(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_and_convert_to_token_props') as \
      load_and_convert_to_token_props_mock:

      valid_response = Mock()
      load_and_convert_to_token_props_mock.side_effect = [
        RuntimeError('oh dear'), valid_response
      ]
      result = load_token_props_list_by_document(
        [FILE_1, FILE_2], None, page_range=PAGE_RANGE, progress=False
      )
      assert result == [valid_response]

class TestTrainModel(object):
  def test_should_train_on_single_file(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_token_props_list_by_document') as \
      load_token_props_list_by_document_mock:
      with patch.object(m, 'CrfSuiteModel') as CrfSuiteModel_mock:
        with patch.object(m, 'pickle') as pickle:
          with patch.object(m, 'token_props_list_to_features') as _:
            train_model([FILE_1], [FILE_2], page_range=PAGE_RANGE, progress=False)
            load_token_props_list_by_document_mock.assert_called_with(
              [FILE_1], [FILE_2], page_range=PAGE_RANGE, progress=False
            )
            model = CrfSuiteModel_mock.return_value
            model.fit.assert_called_with(ANY, ANY)
            pickle.dumps.assert_called_with(model)

  def test_should_raise_error_if_no_documents_have_been_loaded(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_token_props_list_by_document') as \
      load_token_props_list_by_document_mock:
      with patch.object(m, 'CrfSuiteModel'):
        with patch.object(m, 'pickle'):
          with patch.object(m, 'token_props_list_to_features') as _:
            with pytest.raises(AssertionError):
              load_token_props_list_by_document_mock.return_value = []
              train_model([FILE_1], [FILE_2], page_range=PAGE_RANGE)

class TestSaveModel(object):
  def test_should_call_save_content(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'save_file_content') as save_file_content:
      save_model(FILE_1, MODEL_DATA)
      save_file_content.assert_called_with(FILE_1, MODEL_DATA)

class TestRun(object):
  def test_should_train_on_single_file(self):
    m = crfsuite_training_pipeline
    opt = to_namedtuple(
      DEFAULT_ARGS,
      source_file_list=SOURCE_FILE_LIST_PATH,
      output_path=FILE_1,
      limit=2,
      pages=PAGE_RANGE
    )
    with patch.object(m, 'load_file_list') as load_file_list:
      with patch.object(m, 'train_model') as train_model_mock:
        with patch.object(m, 'save_model') as save_model_mock:
          run(opt)
          load_file_list.assert_called_with(
            opt.source_file_list, opt.source_file_column, limit=opt.limit
          )
          train_model_mock.assert_called_with(
            load_file_list.return_value,
            None,
            page_range=PAGE_RANGE
          )
          save_model_mock.assert_called_with(
            opt.output_path,
            train_model_mock.return_value
          )

  def test_should_train_on_single_file_with_cv_output(self):
    m = crfsuite_training_pipeline
    opt = to_namedtuple(
      DEFAULT_ARGS,
      source_file_list=SOURCE_FILE_LIST_PATH,
      cv_source_file_list=CV_SOURCE_FILE_LIST_PATH,
      output_path=FILE_1,
      limit=2,
      pages=PAGE_RANGE
    )
    file_list = [FILE_1, FILE_2]
    cv_file_list = ['cv.' + FILE_1, 'cv.' + FILE_2]
    with patch.object(m, 'load_file_list') as load_file_list:
      with patch.object(m, 'train_model') as train_model_mock:
        with patch.object(m, 'save_model') as save_model_mock:
          load_file_list.side_effect = [file_list, cv_file_list]
          run(opt)
          load_file_list.assert_any_call(
            opt.source_file_list, opt.source_file_column, limit=opt.limit
          )
          load_file_list.assert_any_call(
            opt.cv_source_file_list, opt.cv_source_file_column, limit=opt.limit
          )
          train_model_mock.assert_called_with(
            file_list,
            cv_file_list,
            page_range=PAGE_RANGE
          )
          save_model_mock.assert_called_with(
            opt.output_path,
            train_model_mock.return_value
          )

class TestMain(object):
  def test_should(self):
    argv = ['--source-file-list', SOURCE_FILE_LIST_PATH]
    with patch.object(crfsuite_training_pipeline, 'parse_args') as parse_args_mock:
      with patch.object(crfsuite_training_pipeline, 'run') as run_mock:
        main(argv)
        parse_args_mock.assert_called_with(argv)
        run_mock.assert_called_with(parse_args_mock.return_value)
