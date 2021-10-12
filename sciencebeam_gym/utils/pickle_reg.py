import copyreg
import cv2


# based on: https://stackoverflow.com/a/48832618/8676953
def _pickle_keypoints(point):
    return (
        cv2.KeyPoint,
        (
            *point.pt, point.size, point.angle,
            point.response, point.octave, point.class_id
        )
    )


def register_pickle_functions():
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
