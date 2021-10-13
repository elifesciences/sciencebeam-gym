import pickle

import cv2 as cv

from sciencebeam_gym.utils.pickle_reg import register_pickle_functions


class TestRegisterPickleFunction:
    def test_should_be_able_to_pickle_and_unpickle_cv_keypoint(self):
        register_pickle_functions()
        key_point = cv.KeyPoint(x=1, y=2, size=3, angle=4, response=5, octave=6, class_id=7)
        unpickled_key_point = pickle.loads(pickle.dumps(key_point))
        assert unpickled_key_point.pt == key_point.pt
        assert unpickled_key_point.size == key_point.size
        assert unpickled_key_point.angle == key_point.angle
        assert unpickled_key_point.response == key_point.response
        assert unpickled_key_point.octave == key_point.octave
        assert unpickled_key_point.class_id == key_point.class_id
