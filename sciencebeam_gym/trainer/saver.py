import logging

import tensorflow as tf

from sciencebeam_gym.inference_model import (
  save_inference_model
)

from sciencebeam_gym.trainer.checkpoint import (
  load_last_checkpoint_as_inference_model
)

def get_logger():
  return logging.getLogger(__name__)

def load_checkpoint_and_save_model(model, checkpoint_path, export_dir):
  with tf.Session(graph=tf.Graph()):
    inference_model = load_last_checkpoint_as_inference_model(model, checkpoint_path)
    save_inference_model(export_dir, inference_model)
