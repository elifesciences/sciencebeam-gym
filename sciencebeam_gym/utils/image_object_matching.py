import logging
from typing import List, NamedTuple

import PIL.Image
import numpy as np
from cv2 import cv2 as cv

from sciencebeam_gym.utils.bounding_box import BoundingBox


LOGGER = logging.getLogger(__name__)

FLANN_INDEX_KDTREE = 1


class ObjectDetectorMatcher(NamedTuple):
    detector: cv.Feature2D
    matcher: cv.DescriptorMatcher


def get_sift_detector_matcher(
    flann_tree_count: int = 5,
    flann_check_count: int = 50
) -> ObjectDetectorMatcher:
    detector = cv.SIFT_create()
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_tree_count)
    search_params = dict(checks=flann_check_count)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    return ObjectDetectorMatcher(detector=detector, matcher=matcher)


def to_opencv_image(pil_image: PIL.Image.Image):
    return cv.cvtColor(np.array(pil_image.convert('RGB')), cv.COLOR_RGB2BGR)


def get_bounding_box_for_image(image: PIL.Image.Image) -> BoundingBox:
    return BoundingBox(0, 0, image.width, image.height)


def get_bounding_box_for_points(points: List[List[float]]) -> BoundingBox:
    LOGGER.debug('points: %s', points)
    x_list = [x for x, _ in points]
    y_list = [y for _, y in points]
    x = min(x_list)
    y = min(y_list)
    return BoundingBox(x, y, max(x_list) - x, max(y_list) - y)


def get_object_match(
    object_detector_matcher: ObjectDetectorMatcher,
    target_image: PIL.Image.Image,
    template_image: PIL.Image.Image,
    min_match_count: int = 10,
    knn_cluster_count: int = 2,
    knn_max_distance: float = 0.7,
    ransac_threshold: float = 5.0
):
    detector = object_detector_matcher.detector
    matcher = object_detector_matcher.matcher
    opencv_query_image = to_opencv_image(template_image)
    opencv_train_image = to_opencv_image(target_image)
    kp_query, des_query = detector.detectAndCompute(opencv_query_image, None)
    kp_train, des_train = detector.detectAndCompute(opencv_train_image, None)
    knn_matches = matcher.knnMatch(des_query, des_train, k=knn_cluster_count)
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
    matrix, _mask = cv.findHomography(
        query_pts, train_pts, cv.RANSAC, ransac_threshold
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
    dst = cv.perspectiveTransform(pts, matrix)
    LOGGER.debug('dst: %s', dst)
    return dst
