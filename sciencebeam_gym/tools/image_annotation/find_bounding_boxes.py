import argparse
import logging
from typing import List, Optional

from sciencebeam_gym.tools.image_annotation.find_bounding_boxes_utils import (
    FindBoundingBoxPipelineFactory,
    parse_args,
    process_args
)


LOGGER = logging.getLogger(__name__)


def run(args: argparse.Namespace):
    FindBoundingBoxPipelineFactory(args).run(
        args
    )


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    if args.debug:
        for name in ['__main__', 'sciencebeam_gym']:
            logging.getLogger(name).setLevel(logging.DEBUG)
    LOGGER.info('args: %s', args)
    process_args(args)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
