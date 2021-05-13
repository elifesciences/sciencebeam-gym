import pickle
from pathlib import Path

import pytest

from six import text_type


from lxml import etree
from lxml.builder import E

from sciencebeam_utils.utils.collection import extend_dict

from sciencebeam_gym.models.text.crf.autocut_training_pipeline import (
    parse_args,
    main
)


TITLE_1 = 'The scientific life of mice'
TITLE_2 = 'Cat and mouse'


DEFAULT_ARGS = {
    'input-file-list': 'input.lst',
    'input-xpath': 'title',
    'target-file-list': 'target.lst',
    'target-xpath': 'title',
    'output-path': 'output.pkl'
}


def _get_argv(args):
    return ['--%s=%s' % (key, value) for key, value in args.items()]


def _to_xml(value, tag_name):
    return E.root(
        E(tag_name, value)  # pylint: disable=not-callable
    )


def _write_xml_files(prefix, root_nodes):
    file_list = [
        '%s%d.xml' % (prefix, i) for i in range(len(root_nodes))
    ]
    for root, file_path in zip(root_nodes, file_list):
        with open(file_path, 'wb') as fp:
            etree.ElementTree(root).write(fp)
    return file_list


def _write_xml_files_as_file_list(prefix, root_nodes):
    file_list_path = '%s.lst' % prefix
    file_list = _write_xml_files('%s_' % prefix, root_nodes)
    Path(file_list_path).write_text(text_type('\n'.join(file_list)))
    return file_list_path


class TestParseArgs(object):
    def test_should_parse_namespaces(self):
        opt = parse_args(_get_argv(extend_dict(DEFAULT_ARGS, {
            'namespaces': '{"xyz": "http://xyz"}'
        })))
        assert opt.namespaces == {'xyz': 'http://xyz'}


@pytest.mark.slow
class TestMain(object):
    def test_should_train_end_to_end(self, tmpdir):
        input_titles = ['Title: ' + TITLE_1]
        target_titles = [TITLE_1]
        test_input_titles = ['Title: ' + TITLE_2]
        test_target_titles = [TITLE_2]

        output_path = tmpdir.join('output.pkl')
        input_file_list_path = _write_xml_files_as_file_list(
            tmpdir.join('input'),
            [_to_xml(value, 'title') for value in input_titles]
        )
        target_file_list_path = _write_xml_files_as_file_list(
            tmpdir.join('target'),
            [_to_xml(value, 'title') for value in target_titles]
        )

        main([
            '--input-file-list=%s' % input_file_list_path,
            '--input-xpath=title',
            '--target-file-list=%s' % target_file_list_path,
            '--target-xpath=title',
            '--output-path=%s' % output_path
        ])
        assert output_path.exists()
        model = pickle.loads(output_path.read_binary())
        assert model.predict(test_input_titles) == test_target_titles
