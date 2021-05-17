import argparse
import logging
import re
from collections import Counter
from typing import Iterable, List, Optional

from lxml import etree

import pandas as pd


LOGGER = logging.getLogger(__name__)


# delimters copied from delft/utilities/Tokenizer.py
DELIMITERS = "\n\r\t\f\u00A0([ •*,:;?.!/)-−–‐\"“”‘’'`$]*\u2666\u2665\u2663\u2660\u00A0"
DELIMITERS_REGEX = '(' + '|'.join(map(re.escape, DELIMITERS)) + ')'


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output-word-count-file',
        type=str,
        required=True
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
        etree.parse(xml_file).getroot()
    )


def run(args: argparse.Namespace):
    LOGGER.info('args=%r', args)
    word_counts = Counter(iter_tokens_from_xml_file(args.input_file))
    word_count_df = pd.DataFrame(
        {
            'token': key,
            'count': value
        } for key, value in word_counts.items()
    )
    word_count_df.to_csv(args.output_word_count_file, sep='\t', index=False)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
