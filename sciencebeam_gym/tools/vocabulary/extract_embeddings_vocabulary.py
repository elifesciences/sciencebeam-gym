import argparse
import logging
from typing import Iterable, List, Optional

from tqdm import tqdm

from sciencebeam_gym.utils.io import open_file


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output-vocabulary-file',
        type=str,
        required=True
    )
    return parser.parse_args(argv)


def iter_tokens_from_embeddings_lines(
    lines: Iterable[str]
) -> Iterable[str]:
    for line in lines:
        LOGGER.debug('line: %r', line)
        if not line:
            continue
        token, *_ = line.split(' ', maxsplit=1)
        yield token


def iter_tokens_from_embeddings_file(embeddings_file: str) -> Iterable[str]:
    with open_file(embeddings_file, 'rt') as in_fp:
        yield from iter_tokens_from_embeddings_lines(
            tqdm(in_fp)
        )


def run(args: argparse.Namespace):
    LOGGER.info('args=%r', args)
    tokens_iterable = iter_tokens_from_embeddings_file(
        args.input_file
    )
    with open(args.output_vocabulary_file, 'wt') as out_fp:
        out_fp.writelines((
            token + '\n'
            for token in tokens_iterable
        ))


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
