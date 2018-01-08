from mock import patch, ANY

from sciencebeam_gym.utils.collection import (
  to_namedtuple
)

import sciencebeam_gym.models.text.crf.crfsuite_training_pipeline as crfsuite_training_pipeline
from sciencebeam_gym.models.text.crf.crfsuite_training_pipeline import (
  train_model,
  save_model,
  run,
  main
)

SOURCE_FILE_LIST_PATH = '.temp/source-file-list.lst'

FILE_1 = 'file1.pdf'
FILE_2 = 'file2.pdf'
UNICODE_FILE_1 = u'file1\u1234.pdf'

MODEL_DATA = b'model data'

class TestTrainModel(object):
  def test_should_train_on_single_file(self):
    m = crfsuite_training_pipeline
    with patch.object(m, 'load_structured_document') as _:
      with patch.object(m, 'CrfSuiteModel') as CrfSuiteModel_mock:
        with patch.object(m, 'pickle') as pickle:
          with patch.object(m, 'token_props_list_to_features') as _:
            train_model([FILE_1])
            model = CrfSuiteModel_mock.return_value
            model.fit.assert_called_with(ANY, ANY)
            pickle.dumps.assert_called_with(model)

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
      source_file_list=SOURCE_FILE_LIST_PATH,
      source_file_column='url',
      output_path=FILE_1,
      limit=2
    )
    with patch.object(m, 'load_file_list') as load_file_list:
      with patch.object(m, 'train_model') as train_model_mock:
        with patch.object(m, 'save_model') as save_model_mock:
          run(opt)
          load_file_list.assert_called_with(
            opt.source_file_list, opt.source_file_column, limit=opt.limit
          )
          train_model_mock.assert_called_with(load_file_list.return_value)
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
