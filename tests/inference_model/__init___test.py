import logging
import os
from shutil import rmtree

import tensorflow as tf
import numpy as np

from sciencebeam_utils.utils.num import (
    assert_all_close
)

from sciencebeam_gym.inference_model import (
    InferenceModel,
    save_inference_model,
    load_inference_model
)


TEMP_DIR = '.temp/tests/%s' % __name__

LABELS = [b'label 1', b'label 2', b'label 3']
COLORS = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]


def get_logger():
    return logging.getLogger(__name__)


def setup_module():
    logging.basicConfig(level='DEBUG')


class TestInferenceModelSaverLoader(object):
    def test_should_export_import_and_run_simple_model(self):
        export_dir = os.path.join(TEMP_DIR, 'export')
        if os.path.isdir(export_dir):
            rmtree(export_dir)

        # sample fn that works with tf and np
        def sample_fn(x):
            return x * 2.0 + 10.0

        with tf.Graph().as_default():
            with tf.variable_scope('scope1'):
                inputs = tf.placeholder(tf.float32, (None, 16, 16, 3))
                outputs = sample_fn(inputs)
                get_logger().info('outputs: %s', outputs)
                with tf.Session() as session:
                    save_inference_model(
                        export_dir,
                        InferenceModel(inputs, outputs)
                    )
        with tf.Graph().as_default():
            with tf.Session() as session:
                inference_model = load_inference_model(export_dir)
                inputs_value = np.ones((5, 16, 16, 3))
                assert_all_close(
                    inference_model(inputs_value, session=session),
                    sample_fn(inputs_value)
                )

    def test_should_export_import_color_map(self):
        export_dir = os.path.join(TEMP_DIR, 'export')
        if os.path.isdir(export_dir):
            rmtree(export_dir)

        with tf.Graph().as_default():
            with tf.variable_scope('scope1'):
                inputs = tf.placeholder(tf.float32, (None, 16, 16, 3))
                outputs = inputs * 2.0
                labels = tf.constant(LABELS)
                colors = tf.constant(COLORS)
                with tf.Session():
                    save_inference_model(
                        export_dir,
                        InferenceModel(inputs, outputs, labels, colors)
                    )
        with tf.Graph().as_default():
            with tf.Session():
                inference_model = load_inference_model(export_dir)
                color_map = inference_model.get_color_map()
                get_logger().debug('color_map: %s', color_map)
                assert set(color_map.items()) == set(zip(LABELS, COLORS))
