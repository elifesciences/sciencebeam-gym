import logging
from io import BytesIO

from PIL import Image
import numpy as np
import tensorflow as tf

from sciencebeam_gym.utils.tf import FileIO

from sciencebeam_gym.inference_model import (
    load_inference_model
)

from sciencebeam_gym.trainer.checkpoint import (
    load_last_checkpoint_as_inference_model
)


def get_logger():
    return logging.getLogger(__name__)


def predict_using_inference_model(
        inference_model, predict_filename, output_image_filename):

    with FileIO(predict_filename, 'rb') as input_f:
        image_bytes = input_f.read()
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_data = np.asarray(img, dtype=np.uint8)
        img_data_batch = np.reshape(img_data, tuple([1] + list(img_data.shape)))

    output_img_data_batch = inference_model(img_data_batch)
    output_img_data = output_img_data_batch[0]
    output_image = Image.fromarray(output_img_data, 'RGB')
    out = BytesIO()
    output_image.save(out, 'png')
    output_image_bytes = out.getvalue()

    get_logger().info('writing to %s', output_image_filename)
    with FileIO(output_image_filename, 'wb') as output_f:
        output_f.write(output_image_bytes)


def load_saved_model_and_predict(export_dir, predict_filename, output_image_filename):
    with tf.Session(graph=tf.Graph()):
        predict_using_inference_model(
            load_inference_model(export_dir),
            predict_filename,
            output_image_filename
        )


def load_checkpoint_and_predict(model, checkpoint_path, predict_filename, output_image_filename):
    with tf.Session(graph=tf.Graph()):
        predict_using_inference_model(
            load_last_checkpoint_as_inference_model(model, checkpoint_path),
            predict_filename,
            output_image_filename
        )
