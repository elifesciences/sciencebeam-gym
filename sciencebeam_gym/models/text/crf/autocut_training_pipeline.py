import logging
import argparse
import pickle
import json

from lxml import etree

from apache_beam.io.filesystems import FileSystems

from sciencebeam_utils.beam_utils.io import save_file_content
from sciencebeam_utils.utils.xml import get_text_content
from sciencebeam_utils.utils.file_list import load_file_list

from sciencebeam_gym.models.text.crf.autocut_model import AutocutModel


LOGGER = logging.getLogger(__name__)


def _add_value_args(parser, name):
    parser.add_argument(
        '--%s-file-list' % name, type=str, required=True,
        help='path to %s file list (tsv/csv/lst)' % name
    )
    parser.add_argument(
        '--%s-file-column' % name, type=str, required=False,
        default='url',
        help='csv/tsv column (ignored for plain file list)'
    )
    parser.add_argument(
        '--%s-xpath' % name, type=str, required=True,
        help='xpath to field property, if %s file refers to an XML file' % name
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser('Trains the Autocut model')
    input_parser = parser.add_argument_group('input')
    _add_value_args(input_parser, 'input')

    target_parser = parser.add_argument_group('target')
    _add_value_args(target_parser, 'target')

    output = parser.add_argument_group('output')
    output.add_argument(
        '--output-path', type=str, required=True,
        help='output path to model'
    )

    parser.add_argument(
        '--namespaces', type=json.loads, required=False,
        help='xpath namespaces'
    )

    parser.add_argument(
        '--limit', type=int, required=False,
        help='limit the files to process'
    )

    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='enable debug output'
    )

    return parser.parse_args(argv)


def _load_xml(file_path):
    parser = etree.XMLParser(encoding='utf-8', recover=True)
    try:
        with FileSystems.open(file_path) as fp:
            return etree.parse(fp, parser=parser)
    except Exception as e:
        LOGGER.error('failed to parse: %s due to %s', file_path, e)
        raise


def _extract_value_from_file(file_path, xpath, namespaces):
    root = _load_xml(file_path)
    return '\n'.join(get_text_content(node) for node in root.xpath(xpath, namespaces=namespaces))


def _load_values(file_list_path, file_column, xpath, limit, namespaces):
    file_list = load_file_list(
        file_list_path,
        file_column,
        limit=limit
    )
    return [
        _extract_value_from_file(file_path, xpath, namespaces)
        for file_path in file_list
    ]


def serialize_model(model):
    return pickle.dumps(model)


def save_model(output_filename, model_bytes):
    LOGGER.info('saving model to %s', output_filename)
    save_file_content(output_filename, model_bytes)


def train_model(input_values, target_values):
    model = AutocutModel()
    model.fit(input_values, target_values)
    serialized_model = serialize_model(model)
    return serialized_model


def run(opt):
    input_values = _load_values(
        opt.input_file_list, opt.input_file_column, opt.input_xpath, opt.limit,
        opt.namespaces
    )
    target_values = _load_values(
        opt.target_file_list, opt.target_file_column, opt.target_xpath, opt.limit,
        opt.namespaces
    )
    save_model(
        opt.output_path,
        train_model(input_values, target_values)
    )


def main(argv=None):
    args = parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel('DEBUG')

    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    logging.getLogger('oauth2client').setLevel('WARN')

    main()
