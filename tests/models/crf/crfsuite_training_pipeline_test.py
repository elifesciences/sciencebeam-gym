from mock import patch, Mock, ANY

import pytest

from sciencebeam_utils.utils.collection import (
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
    cv_source_file_column='url',
    cv_source_tag_scope=CV_TAG_SCOPE
)


@pytest.fixture(name='load_structured_document_mock')
def _load_structured_document_mock():
    with patch.object(crfsuite_training_pipeline, 'load_structured_document') as _mock:
        yield _mock


@pytest.fixture(name='structured_document_to_token_props_mock')
def _structured_document_to_token_props_mock():
    with patch.object(crfsuite_training_pipeline, 'structured_document_to_token_props') as _mock:
        yield _mock


@pytest.fixture(name='token_props_list_to_features_mock')
def _token_props_list_to_features_mock():
    with patch.object(crfsuite_training_pipeline, 'token_props_list_to_features') as _mock:
        yield _mock


@pytest.fixture(name='load_token_props_list_by_document_mock')
def _load_token_props_list_by_document_mock():
    with patch.object(crfsuite_training_pipeline, 'load_token_props_list_by_document') as _mock:
        yield _mock


@pytest.fixture(name='load_and_convert_to_token_props_mock')
def _load_and_convert_to_token_props_mock():
    with patch.object(crfsuite_training_pipeline, 'load_and_convert_to_token_props') as _mock:
        yield _mock


@pytest.fixture(name='CrfSuiteModel_mock')
def _CrfSuiteModel_mock():
    with patch.object(crfsuite_training_pipeline, 'CrfSuiteModel') as _mock:
        yield _mock


@pytest.fixture(name='pickle_mock')
def _pickle_mock():
    with patch.object(crfsuite_training_pipeline, 'pickle') as _mock:
        yield _mock


@pytest.fixture(name='save_file_content_mock')
def _save_file_content_mock():
    with patch.object(crfsuite_training_pipeline, 'save_file_content') as _mock:
        yield _mock


@pytest.fixture(name='load_file_list_mock')
def _load_file_list_mock():
    with patch.object(crfsuite_training_pipeline, 'load_file_list') as _mock:
        yield _mock


@pytest.fixture(name='train_model_mock')
def _train_model_mock():
    with patch.object(crfsuite_training_pipeline, 'train_model') as _mock:
        yield _mock


@pytest.fixture(name='save_model_mock')
def _save_model_mock():
    with patch.object(crfsuite_training_pipeline, 'save_model') as _mock:
        yield _mock


class TestLoadAndConvertToTokenProps(object):
    def test_should_load_and_convert_document(
            self,
            load_structured_document_mock,
            structured_document_to_token_props_mock):

        load_and_convert_to_token_props(
            FILE_1, None, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE
        )
        load_structured_document_mock.assert_called_with(
            FILE_1, page_range=PAGE_RANGE
        )
        structured_document_to_token_props_mock.assert_called_with(
            load_structured_document_mock.return_value
        )

    def test_should_load_and_convert_document_with_cv(
            self,
            load_structured_document_mock,
            structured_document_to_token_props_mock):

        load_and_convert_to_token_props(
            FILE_1, FILE_2, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE
        )
        load_structured_document_mock.assert_any_call(
            FILE_1, page_range=PAGE_RANGE
        )
        structured_document_to_token_props_mock.assert_called_with(
            load_structured_document_mock.return_value
        )

    def test_should_merge_doc_and_scope_cv_tag(
            self,
            load_structured_document_mock,
            structured_document_to_token_props_mock):

        structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1, tag=TAG_1)
        ])])
        cv_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1, tag=TAG_2, tag_scope=CV_TAG_SCOPE)
        ])])
        load_structured_document_mock.side_effect = [
            structured_document,
            cv_structured_document
        ]
        load_and_convert_to_token_props(
            FILE_1, FILE_2, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE
        )
        load_structured_document_mock.assert_any_call(
            FILE_1, page_range=PAGE_RANGE
        )
        structured_document_arg = structured_document_to_token_props_mock.call_args[0][0]
        assert [
            structured_document_arg.get_tag_by_scope(t)
            for t in structured_document_arg.iter_all_tokens()
        ] == [{None: TAG_1, CV_TAG_SCOPE: TAG_2}]


class TestLoadTokenPropsListByDocument(object):
    def test_should_load_single_file_without_cv(
            self,
            load_and_convert_to_token_props_mock):

        result = load_token_props_list_by_document(
            [FILE_1], None, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE, progress=False
        )
        load_and_convert_to_token_props_mock.assert_called_with(
            FILE_1, None, cv_source_tag_scope=CV_TAG_SCOPE, page_range=PAGE_RANGE
        )
        assert result == [load_and_convert_to_token_props_mock.return_value]

    def test_should_load_single_file_with_cv(
            self,
            load_and_convert_to_token_props_mock):

        result = load_token_props_list_by_document(
            [FILE_1], [FILE_2], cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE, progress=False
        )
        load_and_convert_to_token_props_mock.assert_called_with(
            FILE_1, FILE_2, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE
        )
        assert result == [load_and_convert_to_token_props_mock.return_value]

    def test_should_load_multiple_files(
            self,
            load_and_convert_to_token_props_mock):

        return_values = [Mock(), Mock()]
        load_and_convert_to_token_props_mock.side_effect = return_values
        result = load_token_props_list_by_document(
            [FILE_1, FILE_2], None, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE, progress=False
        )
        load_and_convert_to_token_props_mock.assert_any_call(
            FILE_1, None, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE
        )
        load_and_convert_to_token_props_mock.assert_any_call(
            FILE_2, None, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE
        )
        assert set(result) == set(return_values)

    def test_should_skip_files_with_errors(
            self,
            load_and_convert_to_token_props_mock):

        valid_response = Mock()
        load_and_convert_to_token_props_mock.side_effect = [
            RuntimeError('oh dear'), valid_response
        ]
        result = load_token_props_list_by_document(
            [FILE_1, FILE_2], None, cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE, progress=False
        )
        assert result == [valid_response]


@pytest.mark.usefixtures(
    'token_props_list_to_features_mock',
    'CrfSuiteModel_mock',
    'pickle_mock'
)
class TestTrainModel(object):
    def test_should_train_on_single_file(
            self,
            load_token_props_list_by_document_mock,
            CrfSuiteModel_mock,
            pickle_mock):

        train_model(
            [FILE_1], [FILE_2],
            cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE, progress=False
        )
        load_token_props_list_by_document_mock.assert_called_with(
            [FILE_1], [FILE_2], cv_source_tag_scope=CV_TAG_SCOPE,
            page_range=PAGE_RANGE, progress=False
        )
        model = CrfSuiteModel_mock.return_value
        model.fit.assert_called_with(ANY, ANY)
        pickle_mock.dumps.assert_called_with(model)

    def test_should_raise_error_if_no_documents_have_been_loaded(
            self,
            load_token_props_list_by_document_mock):

        with pytest.raises(AssertionError):
            load_token_props_list_by_document_mock.return_value = []
            train_model(
                [FILE_1], [FILE_2],
                cv_source_tag_scope=CV_TAG_SCOPE,
                page_range=PAGE_RANGE
            )


class TestSaveModel(object):
    def test_should_call_save_content(
            self,
            save_file_content_mock):
        save_model(FILE_1, MODEL_DATA)
        save_file_content_mock.assert_called_with(FILE_1, MODEL_DATA)


class TestRun(object):
    def test_should_train_on_single_file(
            self,
            load_file_list_mock,
            train_model_mock,
            save_model_mock):

        opt = to_namedtuple(
            DEFAULT_ARGS,
            source_file_list=SOURCE_FILE_LIST_PATH,
            output_path=FILE_1,
            limit=2,
            pages=PAGE_RANGE
        )
        run(opt)
        load_file_list_mock.assert_called_with(
            opt.source_file_list, opt.source_file_column, limit=opt.limit
        )
        train_model_mock.assert_called_with(
            load_file_list_mock.return_value,
            None,
            cv_source_tag_scope=opt.cv_source_tag_scope,
            page_range=PAGE_RANGE
        )
        save_model_mock.assert_called_with(
            opt.output_path,
            train_model_mock.return_value
        )

    def test_should_train_on_single_file_with_cv_output(
            self,
            load_file_list_mock,
            train_model_mock,
            save_model_mock):

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
        load_file_list_mock.side_effect = [file_list, cv_file_list]
        run(opt)
        load_file_list_mock.assert_any_call(
            opt.source_file_list, opt.source_file_column, limit=opt.limit
        )
        load_file_list_mock.assert_any_call(
            opt.cv_source_file_list, opt.cv_source_file_column, limit=opt.limit
        )
        train_model_mock.assert_called_with(
            file_list,
            cv_file_list,
            cv_source_tag_scope=opt.cv_source_tag_scope,
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
