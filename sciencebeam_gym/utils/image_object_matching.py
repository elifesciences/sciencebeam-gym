import logging
from typing import Iterable, List, NamedTuple, Optional

import PIL.Image
import numpy as np
from cv2 import cv2 as cv

from sciencebeam_gym.utils.bounding_box import EMPTY_BOUNDING_BOX, BoundingBox
from sciencebeam_gym.utils.cv import resize_image


LOGGER = logging.getLogger(__name__)

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


class ObjectDetectorMatcher(NamedTuple):
    detector: cv.Feature2D
    matcher: cv.DescriptorMatcher


def get_matcher(
    algorithm: int,
    flann_tree_count: int = 5,
    flann_check_count: int = 50
) -> cv.DescriptorMatcher:
    index_params = dict(algorithm=algorithm, trees=flann_tree_count)
    search_params = dict(checks=flann_check_count)
    return cv.FlannBasedMatcher(index_params, search_params)


def get_sift_detector_matcher(**kwargs) -> ObjectDetectorMatcher:
    detector = cv.SIFT_create()
    matcher = get_matcher(algorithm=FLANN_INDEX_KDTREE, **kwargs)
    return ObjectDetectorMatcher(detector=detector, matcher=matcher)


def get_orb_detector_matcher(**kwargs) -> ObjectDetectorMatcher:
    detector = cv.ORB_create()
    matcher = get_matcher(algorithm=FLANN_INDEX_LSH, **kwargs)
    return ObjectDetectorMatcher(detector=detector, matcher=matcher)


def to_opencv_image(pil_image: PIL.Image.Image):
    return cv.cvtColor(np.array(pil_image.convert('RGB')), cv.COLOR_RGB2BGR)


def get_bounding_box_for_image(image: PIL.Image.Image) -> BoundingBox:
    return BoundingBox(0, 0, image.width, image.height)


def get_bounding_box_for_points(points: List[List[float]]) -> BoundingBox:
    x_list = [x for x, _ in points]
    y_list = [y for _, y in points]
    x = min(x_list)
    y = min(y_list)
    bounding_box = BoundingBox(x, y, max(x_list) - x, max(y_list) - y)
    LOGGER.debug('points: %s, bounding_box: %s', points, bounding_box)
    return bounding_box


def get_filtered_matches(raw_matches: list, max_distance: float) -> list:
    return [
        m[0]
        for m in raw_matches
        if len(m) == 2 and m[0].distance < max_distance * m[1].distance
    ]


class ImageObjectMatchResult(NamedTuple):
    target_points: Optional[np.ndarray]
    keypoint_match_count: int = 0

    def __bool__(self) -> bool:
        return self.target_points is not None

    @property
    def target_bounding_box(self) -> BoundingBox:
        if self.target_points is None:
            return EMPTY_BOUNDING_BOX
        return get_bounding_box_for_points(
            self.target_points.reshape(-1, 2).tolist()
        )


EMPTY_IMAGE_OBJECT_MATCH_RESULT = ImageObjectMatchResult(target_points=None)


def get_image_array_with_max_resolution(
    image_array: np.ndarray,
    max_width: int = 640,
    max_height: int = 480
) -> np.ndarray:
    original_height, original_width = image_array.shape[:2]
    if original_width <= max_width and original_height <= max_height:
        return image_array
    target_width_based_on_height = int(
        original_width * max_height / original_height
    )
    target_height_based_on_width = int(
        original_height * max_width / original_width
    )
    if target_width_based_on_height <= max_width:
        return resize_image(
            image_array, width=target_width_based_on_height, height=max_height
        )
    return resize_image(
        image_array, width=max_width, height=target_height_based_on_width
    )


def _get_resized_opencv_image(
    image: PIL.Image.Image,
    image_cache: dict
) -> np.ndarray:
    opencv_image = image_cache.get(id(image))
    if opencv_image is None:
        opencv_image = cv.cvtColor(
            get_image_array_with_max_resolution(
                to_opencv_image(image)
            ),
            cv.COLOR_BGR2GRAY
        )
        image_cache[id(image)] = opencv_image
    return opencv_image


def get_object_match(
    target_image: PIL.Image.Image,
    template_image: PIL.Image.Image,
    object_detector_matcher: ObjectDetectorMatcher,
    min_match_count: int = 10,
    knn_cluster_count: int = 2,
    knn_max_distance: float = 0.7,
    ransac_threshold: float = 5.0,
    image_cache: Optional[dict] = None
) -> ImageObjectMatchResult:
    if image_cache is None:
        image_cache = {}
    detector = object_detector_matcher.detector
    matcher = object_detector_matcher.matcher
    opencv_query_image = _get_resized_opencv_image(
        template_image,
        image_cache=image_cache
    )
    opencv_train_image = _get_resized_opencv_image(
        target_image,
        image_cache=image_cache
    )
    fx = target_image.width / opencv_train_image.shape[1]
    fy = target_image.height / opencv_train_image.shape[0]
    kp_query, des_query = detector.detectAndCompute(opencv_query_image, None)
    kp_train, des_train = detector.detectAndCompute(opencv_train_image, None)
    if des_train is None:
        LOGGER.debug('no keypoints found in target image (train)')
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
    if des_query is None:
        LOGGER.debug('no keypoints found in template image (query)')
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
    LOGGER.debug('des_query: %s', des_query)
    LOGGER.debug('des_train: %s', des_train)
    knn_matches = matcher.knnMatch(des_query, des_train, k=knn_cluster_count)
    good_matches = get_filtered_matches(knn_matches, knn_max_distance)
    LOGGER.debug('good_matches: %d (%s...)', len(good_matches), good_matches[:3])
    LOGGER.debug(
        'good_matches[:3].pt: %s...', [kp_query[m.queryIdx].pt for m in good_matches[:3]]
    )
    if len(good_matches) < min_match_count:
        LOGGER.debug('not enough matches')
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
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
    if matrix is None:
        LOGGER.debug('no homography found')
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
    h, w = opencv_query_image.shape[:2]
    LOGGER.debug('w: %s, h: %s', w, h)
    pts = np.array([
        [[0, 0]],
        [[0, h]],
        [[w, h]],
        [[w, 0]]
    ], dtype=np.float32)
    LOGGER.debug('pts: %s', pts)
    dst = cv.perspectiveTransform(pts, matrix) * [fx, fy]
    LOGGER.debug('dst: %s', dst)
    return ImageObjectMatchResult(
        target_points=dst,
        keypoint_match_count=len(good_matches)
    )


class ImageListObjectMatchResult(NamedTuple):
    target_image_index: int
    match_result: ImageObjectMatchResult

    def __bool__(self) -> bool:
        return self.target_image_index >= 0

    @property
    def target_bounding_box(self) -> BoundingBox:
        return self.match_result.target_bounding_box


EMPTY_IMAGE_LIST_OBJECT_MATCH_RESULT = ImageListObjectMatchResult(
    target_image_index=-1,
    match_result=EMPTY_IMAGE_OBJECT_MATCH_RESULT
)


def iter_image_list_object_match(
    target_images: List[np.ndarray],
    *args,
    **kwargs
) -> Iterable[ImageListObjectMatchResult]:
    for target_image_index, target_image in enumerate(target_images):
        match_result = get_object_match(target_image, *args, **kwargs)
        if not match_result:
            continue
        yield ImageListObjectMatchResult(
            target_image_index=target_image_index,
            match_result=match_result
        )


def get_image_list_object_match(
    *args,
    **kwargs
) -> ImageListObjectMatchResult:
    best_image_list_object_match = EMPTY_IMAGE_LIST_OBJECT_MATCH_RESULT
    for image_list_object_match in iter_image_list_object_match(*args, **kwargs):
        LOGGER.debug(
            'image_list_object_match.match_result.keypoint_match_count: %s',
            image_list_object_match.match_result.keypoint_match_count
        )
        if (
            image_list_object_match.match_result.keypoint_match_count
            > best_image_list_object_match.match_result.keypoint_match_count
        ):
            best_image_list_object_match = image_list_object_match
    return best_image_list_object_match
