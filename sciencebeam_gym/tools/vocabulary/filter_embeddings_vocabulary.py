import argparse
import logging
from typing import Iterable, List, Optional, Set

from tqdm import tqdm
import pandas as pd


from sciencebeam_gym.utils.io import open_file


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-embeddings-file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--word-count-file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output-embeddings-file',
        type=str,
        required=True
    )
    return parser.parse_args(argv)


def read_vocabulary_set_from_word_count_file(
    word_count_file: str
) -> Set[str]:
    return set(pd.read_csv(word_count_file, sep='\t')['token'].dropna())


def iter_read_lines_from_embeddings_file(
    embeddings_file: str
) -> Iterable[str]:
    with open_file(embeddings_file, 'rt') as in_fp:
        for line in tqdm(in_fp):
            if not line:
                continue
            yield line


def write_lines_to_embeddings_file(
    embeddings_file: str,
    lines: Iterable[str]
):
    with open_file(embeddings_file, 'wt') as out_fp:
        out_fp.writelines((
            line.rstrip() + '\n'
            for line in lines
        ))


def iter_filter_embeddings_lines(
    embeddings_lines: Iterable[str],
    vocabulary: Set[str]
) -> Iterable[str]:
    LOGGER.info('vocabulary: %r', list(vocabulary)[:10])
    vocabulary = {token.lower() for token in vocabulary}
    excluded_tokens = []
    excluded_token_count = 0
    included_token_count = 0
    for line in embeddings_lines:
        LOGGER.debug('line: %r', line)
        token, *_ = line.split(' ', maxsplit=1)
        if token.lower() in vocabulary:
            included_token_count += 1
            yield line
        else:
            excluded_token_count += 1
            if len(excluded_tokens) < 100:
                excluded_tokens.append(token)
    LOGGER.info(
        'filtered tokens, included=%d, excluded=%d (e.g. %r)',
        included_token_count, excluded_token_count, excluded_tokens
    )


def run(
    input_embeddings_file: str,
    word_count_file: str,
    output_embeddings_file: str
):
    vocabulary = read_vocabulary_set_from_word_count_file(
        word_count_file
    )
    LOGGER.info('vocabulary size: %d', len(vocabulary))
    embeddings_lines_iterable = iter_read_lines_from_embeddings_file(
        input_embeddings_file
    )
    filter_embeddings_lines_iterable = iter_filter_embeddings_lines(
        embeddings_lines_iterable,
        vocabulary
    )
    write_lines_to_embeddings_file(
        output_embeddings_file,
        filter_embeddings_lines_iterable
    )


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    run(
        input_embeddings_file=args.input_embeddings_file,
        word_count_file=args.word_count_file,
        output_embeddings_file=args.output_embeddings_file
    )


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
