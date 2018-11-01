import logging

import tensorflow as tf

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_model import (
    batch_dimensions_to_most_likely_colors_list
)

from sciencebeam_gym.inference_model import (
    InferenceModel
)


def get_logger():
    return logging.getLogger(__name__)


def load_last_checkpoint_as_inference_model(model, checkpoint_path, session=None):
    if session is None:
        session = tf.get_default_session()
    assert session is not None
    last_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    get_logger().info(
        'last_checkpoint: %s (%s)', last_checkpoint, checkpoint_path
    )
    tensors = model.build_predict_graph()
    inputs_tensor = tensors.inputs['image']
    outputs_tensor = tensors.pred
    if model.use_separate_channels:
        outputs_tensor = batch_dimensions_to_most_likely_colors_list(
            outputs_tensor,
            model.dimension_colors_with_unknown
        )
    saver = tf.train.Saver()
    saver.restore(session, last_checkpoint)
    labels = tf.constant(model.dimension_labels_with_unknown)
    colors = tf.constant(model.dimension_colors_with_unknown)
    inference_model = InferenceModel(
        inputs_tensor, outputs_tensor,
        labels, colors
    )
    return inference_model
