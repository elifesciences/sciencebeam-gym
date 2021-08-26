import argparse
import logging
from typing import List, Optional


LOGGER = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pdf-file',
        type=str,
        required=True,
        help='Path to the PDF file'
    )
    parser.add_argument(
        '--image-file',
        type=str,
        required=True,
        help='Path to the image to find the bounding boxes for'
    )
    parser.add_argument(
        '--output-json-file',
        required=True,
        type=str,
        help='The path to the output JSON file to write the bounding boxes to.'
    )
    return parser


def parse_args(argv=None):
    parser = get_args_parser()
    parsed_args, _ = parser.parse_known_args(argv)
    return parsed_args


def main(argv=Optional[List[str]]):
    LOGGER.info('argv: %s', argv)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
