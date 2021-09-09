import logging
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import PIL.Image
import numpy as np
from cv2 import cv2 as cv
import skimage.metrics

from sciencebeam_gym.utils.bounding_box import EMPTY_BOUNDING_BOX, BoundingBox
from sciencebeam_gym.utils.cv import (
    crop_image_to_bounding_box,
    resize_image,
    to_opencv_image
)


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
    score: float = 0
    target_bounding_box: BoundingBox = EMPTY_BOUNDING_BOX

    def __bool__(self) -> bool:
        return self.target_points is not None

    @property
    def sort_key(self) -> Tuple[float, int]:
        return (self.score, self.keypoint_match_count)

    @property
    def calculated_target_bounding_box(self) -> BoundingBox:
        if self.target_points is None:
            return EMPTY_BOUNDING_BOX
        return get_bounding_box_for_points(
            self.target_points.reshape(-1, 2).tolist()
        )


EMPTY_IMAGE_OBJECT_MATCH_RESULT = ImageObjectMatchResult(target_points=None)


DEFAULT_MAX_WIDTH = 0
DEFAULT_MAX_HEIGHT = DEFAULT_MAX_WIDTH

DEFAULT_MAX_BOUNDING_BOX_ADJUSTMENT_ITERATIONS = 0


def get_image_array_with_max_resolution(
    image_array: np.ndarray,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT
) -> np.ndarray:
    original_height, original_width = image_array.shape[:2]
    if (
        (not max_width or original_width <= max_width)
        and (not max_height or original_height <= max_height)
    ):
        LOGGER.debug(
            'image within expected dimension: %sx%s <= %sx%s',
            original_width, original_height, max_width, max_height
        )
        return image_array
    target_width_based_on_height = int(
        original_width * max_height / original_height
    )
    target_height_based_on_width = int(
        original_height * max_width / original_width
    )
    if max_height and (not max_width or target_width_based_on_height <= max_width):
        return resize_image(
            image_array, width=target_width_based_on_height, height=max_height
        )
    return resize_image(
        image_array, width=max_width, height=target_height_based_on_width
    )


def _get_resized_opencv_image(
    image: PIL.Image.Image,
    image_cache: dict,
    max_width: int,
    max_height: int,
    use_grayscale: bool,
    image_id: str,
    bounding_box: Optional[BoundingBox] = None
) -> np.ndarray:
    key = f'image-{image_id}-{max_width}-{max_height}-{use_grayscale}'
    if bounding_box:
        key += f'-bbox-{bounding_box}'
    opencv_image = image_cache.get(key)
    if opencv_image is None:
        opencv_image = to_opencv_image(image)
        if bounding_box:
            bounding_box = bounding_box.round().intersection(
                BoundingBox(0, 0, opencv_image.shape[1], opencv_image.shape[0])
            )
            opencv_image = crop_image_to_bounding_box(
                opencv_image,
                bounding_box
            )
        opencv_image = get_image_array_with_max_resolution(
            opencv_image,
            max_width=max_width,
            max_height=max_height
        )
        if use_grayscale:
            opencv_image = cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
        image_cache[key] = opencv_image
    return opencv_image


def _get_detect_and_computed_keypoints(
    image_array: np.ndarray,
    detector: cv.Feature2D,
    image_cache: dict,
    image_id: str
) -> Tuple[Any, Any]:
    key = f'features-{image_id}'
    result = image_cache.get(key)
    if result is None:
        result = detector.detectAndCompute(image_array, None)
        image_cache[key] = result
    return result


class BoundingBoxScoreSummary(NamedTuple):
    score: float
    target_bounding_box: BoundingBox


def _move_bounding_box_edge(
    bounding_box: BoundingBox,
    edge_index: int,
    delta: int
) -> BoundingBox:
    if edge_index == 0:
        return BoundingBox(
            bounding_box.x + delta,
            bounding_box.y,
            bounding_box.width - delta,
            bounding_box.height
        )
    if edge_index == 1:
        return BoundingBox(
            bounding_box.x,
            bounding_box.y + delta,
            bounding_box.width,
            bounding_box.height - delta
        )
    if edge_index == 2:
        return BoundingBox(
            bounding_box.x,
            bounding_box.y,
            bounding_box.width + delta,
            bounding_box.height
        )
    if edge_index == 3:
        return BoundingBox(
            bounding_box.x,
            bounding_box.y,
            bounding_box.width,
            bounding_box.height + delta
        )
    raise RuntimeError(f'invalid edge index: {edge_index}')


def get_bounding_box_match_score_summary(
    target_bounding_box: BoundingBox,
    target_image: PIL.Image.Image,
    template_image: PIL.Image.Image,
    image_cache: dict,
    target_image_id: str,
    template_image_id: str,
    similarity_width: int = 512,  # use fixed similarity size for more consistent score
    similarity_height: int = 512,
    max_bounding_box_adjustment_iterations: int = DEFAULT_MAX_BOUNDING_BOX_ADJUSTMENT_ITERATIONS
) -> BoundingBoxScoreSummary:
    opencv_target_image = _get_resized_opencv_image(
        target_image,
        image_cache=image_cache,
        max_width=0,
        max_height=0,
        use_grayscale=True,
        image_id=target_image_id
    )
    opencv_template_image = _get_resized_opencv_image(
        template_image,
        image_cache=image_cache,
        max_width=0,
        max_height=0,
        use_grayscale=True,
        image_id=template_image_id
    )
    LOGGER.debug('opencv_target_image.shape: %s', opencv_target_image.shape)
    fx = target_image.width / opencv_target_image.shape[1]
    fy = target_image.height / opencv_target_image.shape[0]
    max_bounding_box = BoundingBox(
        0, 0, opencv_target_image.shape[1], opencv_target_image.shape[0]
    )
    bounding_box = (
        target_bounding_box
        .scale_by(1 / fx, 1 / fy)
        .round()
        .intersection(max_bounding_box)
    )
    LOGGER.debug('bounding_box (for score): %s', bounding_box)
    if bounding_box.width < 10 or bounding_box.height < 10:
        LOGGER.debug('bounding box too small')
        return BoundingBoxScoreSummary(score=0.0, target_bounding_box=target_bounding_box)
    resized_template_image = resize_image(
        opencv_template_image,
        width=similarity_width,
        height=similarity_height
    )
    _original_bounding_box = bounding_box
    best_bounding_box = bounding_box
    best_score = 0.0
    previous_bounding_box = bounding_box
    previous_score: float = 0.0
    previous_value_index: int = 0
    next_value_index: int = 0
    directions = [-1, -1, 1, 1]
    score_by_bounding_box: Dict[BoundingBox, float] = {}
    for _ in range(1 + max_bounding_box_adjustment_iterations):
        score = score_by_bounding_box.get(bounding_box)
        if score is None:
            cropped_target_image = crop_image_to_bounding_box(
                opencv_target_image, bounding_box
            )
            LOGGER.debug('cropped_target_image.shape: %s', cropped_target_image.shape)
            score = skimage.metrics.structural_similarity(
                resize_image(cropped_target_image, similarity_width, similarity_height),
                resized_template_image,
            )
            LOGGER.debug('score: %s', score)
            score_by_bounding_box[bounding_box] = score
        if score > best_score:
            best_score = score
            best_bounding_box = bounding_box
        if score >= 0.99:
            break
        if score < previous_score:
            bounding_box = previous_bounding_box
            directions[previous_value_index] = (-1) * directions[previous_value_index]
        previous_bounding_box = bounding_box
        bounding_box = _move_bounding_box_edge(
            bounding_box, next_value_index, directions[next_value_index]
        ).intersection(max_bounding_box)
        previous_value_index = next_value_index
        next_value_index = (next_value_index + 1) % 4
        if not bounding_box or bounding_box == previous_bounding_box:
            bounding_box = previous_bounding_box
            directions[previous_value_index] = (-1) * directions[previous_value_index]
    if best_bounding_box != _original_bounding_box:
        target_bounding_box = (
            best_bounding_box
            .scale_by(fx, fy)
            .round()
            .intersection(get_bounding_box_for_image(target_image))
        )
    return BoundingBoxScoreSummary(
        score=best_score,
        target_bounding_box=target_bounding_box
    )


def _get_object_match(  # pylint: disable=too-many-return-statements
    target_image: PIL.Image.Image,
    template_image: PIL.Image.Image,
    object_detector_matcher: ObjectDetectorMatcher,
    target_image_id: str,
    template_image_id: str,
    image_cache: dict,
    min_match_count: int = 10,
    knn_cluster_count: int = 2,
    knn_max_distance: float = 0.7,
    ransac_threshold: float = 5.0,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_height: int = DEFAULT_MAX_HEIGHT,
    use_grayscale: bool = False,
    score_threshold: float = 0.0,  # using no score threshold for now,
    target_bounding_box: Optional[BoundingBox] = None,
    max_bounding_box_adjustment_iterations: int = DEFAULT_MAX_BOUNDING_BOX_ADJUSTMENT_ITERATIONS
) -> ImageObjectMatchResult:
    detector = object_detector_matcher.detector
    matcher = object_detector_matcher.matcher
    opencv_query_image = _get_resized_opencv_image(
        template_image,
        image_cache=image_cache,
        max_width=max_width,
        max_height=max_height,
        use_grayscale=use_grayscale,
        image_id=template_image_id
    )
    opencv_train_image = _get_resized_opencv_image(
        target_image,
        image_cache=image_cache,
        max_width=max_width,
        max_height=max_height,
        use_grayscale=use_grayscale,
        image_id=target_image_id
    )
    fx = target_image.width / opencv_train_image.shape[1]
    fy = target_image.height / opencv_train_image.shape[0]
    dx, dy = 0, 0
    target_keypoint_id_suffix = ''
    if target_bounding_box:
        scaled_down_target_bounding_box = (
            target_bounding_box
            .scale_by(1 / fx, 1 / fy)
            .round()
        )
        LOGGER.debug(
            'scaled_down_target_bounding_box: %r (orignal: %r)',
            scaled_down_target_bounding_box,
            target_bounding_box
        )
        opencv_train_image = crop_image_to_bounding_box(
            opencv_train_image,
            scaled_down_target_bounding_box
        )
        dx, dy = (int(target_bounding_box.x), int(target_bounding_box.y))
        target_keypoint_id_suffix += f'-bbox-{target_bounding_box}'
    LOGGER.debug('fx=%s, fy=%s, dx=%s, dy=%s', fx, fy, dx, dy)
    kp_query, des_query = _get_detect_and_computed_keypoints(
        opencv_query_image,
        detector=detector,
        image_cache=image_cache,
        image_id=template_image_id
    )
    kp_train, des_train = _get_detect_and_computed_keypoints(
        opencv_train_image,
        detector=detector,
        image_cache=image_cache,
        image_id=target_image_id + target_keypoint_id_suffix
    )
    if des_train is None:
        LOGGER.debug('no keypoints found in target image (train)')
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
    if des_query is None:
        LOGGER.debug('no keypoints found in template image (query)')
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
    LOGGER.debug('des_query (%d): %s', len(des_query), des_query)
    LOGGER.debug('des_train (%d): %s', len(des_train), des_train)
    if len(des_query) < knn_cluster_count or len(des_train) < knn_cluster_count:
        LOGGER.debug('need at least %d keypoints', knn_cluster_count)
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
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
    dst = cv.perspectiveTransform(pts, matrix).reshape(-1, 2)
    LOGGER.debug('dst (internal): %s', dst)
    dst_rescaled = dst * [fx, fy] + [dx, dy]
    LOGGER.debug('dst_rescaled: %s', dst_rescaled)
    _target_bounding_box = get_bounding_box_for_points(dst_rescaled)
    LOGGER.debug('_target_bounding_box: %s', _target_bounding_box)
    score_summary = get_bounding_box_match_score_summary(
        _target_bounding_box,
        target_image=target_image,
        template_image=template_image,
        image_cache=image_cache,
        target_image_id=target_image_id,
        template_image_id=template_image_id,
        max_bounding_box_adjustment_iterations=max_bounding_box_adjustment_iterations
    )
    score = score_summary.score
    LOGGER.debug('score: %s', score)
    if score < score_threshold:
        return EMPTY_IMAGE_OBJECT_MATCH_RESULT
    result = ImageObjectMatchResult(
        target_points=dst_rescaled,
        keypoint_match_count=len(good_matches),
        score=score,
        target_bounding_box=score_summary.target_bounding_box
    )
    return result


def get_object_match(
    target_image: PIL.Image.Image,
    template_image: PIL.Image.Image,
    target_image_id: Optional[str] = None,
    template_image_id: Optional[str] = None,
    image_cache: Optional[dict] = None,
    **kwargs
) -> ImageObjectMatchResult:
    if image_cache is None:
        image_cache = {}
    if not target_image_id:
        target_image_id = str(id(target_image))
    if not template_image_id:
        template_image_id = str(id(template_image))
    result = _get_object_match(
        target_image=target_image,
        target_image_id=target_image_id,
        template_image=template_image,
        template_image_id=template_image_id,
        image_cache=image_cache,
        **kwargs
    )
    if not result:
        return result
    target_bounding_box = (
        result
        .target_bounding_box
        .round()
        .intersection(get_bounding_box_for_image(target_image))
    )
    if target_bounding_box.width < 10 or target_bounding_box.height < 10:
        return result
    LOGGER.debug('finding bounding box within target bbox: %r', target_bounding_box)
    revised_result = _get_object_match(
        target_image=target_image,
        target_image_id=target_image_id,
        template_image=template_image,
        template_image_id=template_image_id,
        image_cache=image_cache,
        target_bounding_box=target_bounding_box,
        **kwargs
    )
    if revised_result.score > result.score:
        LOGGER.debug(
            'score has improved from %s to %s, use revised result',
            result.score, revised_result.score
        )
        result = revised_result
    return result


class ImageListObjectMatchResult(NamedTuple):
    target_image_index: int
    match_result: ImageObjectMatchResult

    def __bool__(self) -> bool:
        return self.target_image_index >= 0

    @property
    def target_bounding_box(self) -> BoundingBox:
        return self.match_result.target_bounding_box

    @property
    def score(self) -> float:
        return self.match_result.score


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
        LOGGER.debug(
            'processing target image: %d / %d',
            1 + target_image_index, len(target_images)
        )
        match_result = get_object_match(  # type: ignore
            target_image,
            *args,
            target_image_id=f'page-{1 + target_image_index}',
            **kwargs
        )
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
            'image_list_object_match.match_result.sort_key: %s',
            image_list_object_match.match_result.sort_key
        )
        if (
            image_list_object_match.match_result.sort_key
            > best_image_list_object_match.match_result.sort_key
        ):
            best_image_list_object_match = image_list_object_match
    return best_image_list_object_match
