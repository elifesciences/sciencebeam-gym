import argparse
import json
import logging
import os
from typing import List, Optional

import PIL.Image
from pdf2image import convert_from_bytes

from sciencebeam_gym.utils.io import read_bytes, write_text


LOGGER = logging.getLogger(__name__)


def get_images_from_pdf(pdf_path: str) -> List[PIL.Image.Image]:
    return convert_from_bytes(read_bytes(pdf_path))


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


def parse_args(argv: Optional[str] = None):
    parser = get_args_parser()
    parsed_args, _ = parser.parse_known_args(argv)
    return parsed_args


def run(pdf_path: str, json_path: str):
    pdf_images = get_images_from_pdf(pdf_path)
    data_json = {
        'images': [
            {
                'file_name': os.path.basename(pdf_path) + '/page_%05d.jpg' % (1 + page_index),
                'width': pdf_image.width,
                'height': pdf_image.height,
                'id': (1 + page_index)
            }
            for page_index, pdf_image in enumerate(pdf_images)
        ],
        'categories': [{
            'id': 1,
            'name': 'figure'
        }]
    }
    write_text(json_path, json.dumps(data_json))


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    LOGGER.info('args: %s', args)
    run(
        pdf_path=args.pdf_file,
        json_path=args.output_json_file
    )


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
