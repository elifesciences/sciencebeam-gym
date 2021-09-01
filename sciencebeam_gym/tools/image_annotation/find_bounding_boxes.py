import argparse
import json
import logging
import os
from io import BytesIO
from typing import List, Optional

import PIL.Image
import numpy as np
from cv2 import cv2
from pdf2image import convert_from_bytes

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.io import read_bytes, write_text


LOGGER = logging.getLogger(__name__)


FLANN_INDEX_KDTREE = 1


def get_images_from_pdf(pdf_path: str) -> List[PIL.Image.Image]:
    return convert_from_bytes(read_bytes(pdf_path))


def to_opencv_image(pil_image: PIL.Image.Image):
    return cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)


def get_bounding_box_for_image(image: PIL.Image.Image) -> BoundingBox:
    return BoundingBox(0, 0, image.width, image.height)


def get_bounding_box_for_points(points: List[List[float]]) -> BoundingBox:
    LOGGER.debug('points: %s', points)
    x_list = [x for x, _ in points]
    y_list = [y for _, y in points]
    x = min(x_list)
    y = min(y_list)
    return BoundingBox(x, y, max(x_list) - x, max(y_list) - y)


def get_sift_match(
    target_image: PIL.Image.Image,
    template_image: PIL.Image.Image,
    min_match_count: int = 10,
    flann_tree_count: int = 5,
    flann_check_count: int = 50,
    knn_cluster_count: int = 2,
    knn_max_distance: float = 0.7,
    ransac_threshold: float = 5.0
):
    sift = cv2.SIFT_create()
    opencv_query_image = to_opencv_image(template_image)
    opencv_train_image = to_opencv_image(target_image)
    kp_query, des_query = sift.detectAndCompute(opencv_query_image, None)
    kp_train, des_train = sift.detectAndCompute(opencv_train_image, None)
    index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': flann_tree_count}
    search_params = {'checks': flann_check_count}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(des_query, des_train, k=knn_cluster_count)
    good_knn_matches = [
        (m, n)
        for m, n in knn_matches
        if m.distance <= knn_max_distance * n.distance
    ]
    good_matches = [m for m, _ in good_knn_matches]
    LOGGER.debug('good_matches: %d (%s...)', len(good_matches), good_matches[:3])
    LOGGER.debug(
        'good_matches[:3].pt: %s...', [kp_query[m.queryIdx].pt for m in good_matches[:3]]
    )
    if len(good_matches) < min_match_count:
        LOGGER.debug('not enough matches')
        return None
    query_pts = np.array([
        [kp_query[m.queryIdx].pt] for m in good_matches
    ], dtype=np.float32)
    LOGGER.debug('query_pts: %d (%s)', len(query_pts), query_pts[:10])
    train_pts = np.array([
        [kp_train[m.trainIdx].pt] for m in good_matches
    ], dtype=np.float32)
    LOGGER.debug('train_pts: %d (%s)', len(train_pts), train_pts[:10])
    matrix, _mask = cv2.findHomography(
        query_pts, train_pts, cv2.RANSAC, ransac_threshold
    )
    LOGGER.debug('matrix: %s', matrix)
    h, w = opencv_query_image.shape[:2]
    LOGGER.debug('w: %s, h: %s', w, h)
    pts = np.array([
        [[0, 0]],
        [[0, h]],
        [[w, h]],
        [[w, 0]]
    ], dtype=np.float32)
    LOGGER.debug('pts: %s', pts)
    dst = cv2.perspectiveTransform(pts, matrix)
    LOGGER.debug('dst: %s', dst)
    return dst


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pdf-file',
        type=str,
        required=True,
        help='Path to the PDF file'
    )
    parser.add_argument(
        '--image-files',
        nargs='+',
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


def parse_args(argv: Optional[List[str]] = None):
    parser = get_args_parser()
    parsed_args, _ = parser.parse_known_args(argv)
    return parsed_args


def run(pdf_path: str, image_paths: List[str], json_path: str):
    pdf_images = get_images_from_pdf(pdf_path)
    for image_path in image_paths:
        template_image = PIL.Image.open(BytesIO(read_bytes(image_path)))
        LOGGER.debug('template_image: %s x %s', template_image.width, template_image.height)
        annotations = []
        for page_index, pdf_image in enumerate(pdf_images):
            pdf_page_bounding_box = get_bounding_box_for_image(pdf_image)
            sift_match = get_sift_match(pdf_image, template_image)
            if sift_match is not None:
                LOGGER.debug('sift_match: %s', sift_match)
                annotations.append({
                    'image_id': (1 + page_index),
                    'category_id': 1,
                    'bbox': get_bounding_box_for_points(
                        sift_match.reshape(-1, 2).tolist()
                    ).intersection(pdf_page_bounding_box).to_list()
                })
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
        'annotations': annotations,
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
        image_paths=args.image_files,
        json_path=args.output_json_file
    )


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
