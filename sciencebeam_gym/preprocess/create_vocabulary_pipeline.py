import argparse
import logging
import re
from collections import Counter
from typing import Iterable, List, Optional

from lxml import etree
from tqdm import tqdm

import pandas as pd

from sciencebeam_utils.beam_utils.io import read_all_from_path
from sciencebeam_utils.utils.file_list import load_file_list


LOGGER = logging.getLogger(__name__)


# delimters copied from delft/utilities/Tokenizer.py
DELIMITERS = "\n\r\t\f\u00A0([ •*,:;?.!/)-−–‐\"“”‘’'`$]*\u2666\u2665\u2663\u2660\u00A0"
DELIMITERS_REGEX = '(' + '|'.join(map(re.escape, DELIMITERS)) + ')'


def _add_file_list_args(
    parser: argparse.ArgumentParser,
    name: str
):
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument(
        f'--{name}-file',
        type=str,
    )
    mutex_group.add_argument(
        f'--{name}-file-list', type=str,
        help=f'path to {name} file list (tsv/csv/lst)'
    )
    parser.add_argument(
        f'--{name}-file-column', type=str, required=False,
        default='url',
        help='csv/tsv column (ignored for plain file list)'
    )


def _get_file_list(
    file_path: str,
    file_list_path: str,
    file_column: str,
    limit: Optional[int] = None
) -> List[str]:
    if file_path:
        return [file_path]
    return load_file_list(
        file_list_path,
        file_column,
        limit=limit
    )


def get_input_file_list_from_args(
    args: argparse.Namespace,
    limit: Optional[int] = None
) -> List[str]:
    return _get_file_list(
        args.input_file,
        args.input_file_list,
        args.input_file_column,
        limit=limit
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_parser = parser.add_argument_group('input')
    _add_file_list_args(input_parser, 'input')
    input_parser.add_argument(
        '--limit',
        type=int,
        required=False
    )

    parser.add_argument(
        '--output-word-count-file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--sort-by-count',
        action='store_true'
    )
    return parser.parse_args(argv)


def iter_tokenized_tokens(text: str) -> List[str]:
    for token in re.split(DELIMITERS_REGEX, text.strip()):
        if not token.strip():
            continue
        yield token


def iter_tokens_from_xml_root(
    xml_root: etree.ElementBase
) -> Iterable[str]:
    for text in xml_root.itertext():
        yield from iter_tokenized_tokens(text)


def iter_tokens_from_xml_file(xml_file: str) -> Iterable[str]:
    yield from iter_tokens_from_xml_root(
        etree.fromstring(
            read_all_from_path(xml_file)
        )
    )


def iter_tokens_from_xml_file_list(
    xml_file_list: Iterable[str]
) -> Iterable[str]:
    for xml_file in xml_file_list:
        yield from iter_tokens_from_xml_file(xml_file)


def run(
    input_file_list: List[str],
    output_word_count_file: str,
    sort_by_count: bool
):
    flat_tokens_iterable = iter_tokens_from_xml_file_list(
        tqdm(input_file_list)
    )
    word_counts = Counter(flat_tokens_iterable)
    word_count_df = pd.DataFrame(
        {
            'token': key,
            'count': value
        } for key, value in word_counts.items()
    )
    if sort_by_count:
        word_count_df = word_count_df.sort_values('token', ascending=True)
        word_count_df = word_count_df.sort_values('count', ascending=False)
    word_count_df.to_csv(output_word_count_file, sep='\t', index=False)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    LOGGER.info('args=%r', args)
    input_file_list = get_input_file_list_from_args(args, limit=args.limit)
    LOGGER.debug('input_file_list: %s', input_file_list)
    run(
        input_file_list=input_file_list,
        output_word_count_file=args.output_word_count_file,
        sort_by_count=args.sort_by_count
    )


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
