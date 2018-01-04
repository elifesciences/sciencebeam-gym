import logging

import tensorflow as tf

# pylint: disable=E0611
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.lib.io.file_io import delete_recursively, is_directory
# pylint: enable=E0611

INPUTS_KEY = 'images'
OUTPUTS_KEY = 'annotation'
LABELS_KEY = 'labels'
COLORS_KEY = 'colors'

def get_logger():
  return logging.getLogger(__name__)

class InferenceModel(object):
  def __init__(self, inputs, outputs, labels_tensor=None, colors_tensor=None):
    self.inputs_tensor = inputs
    self.outputs_tensor = outputs
    self.labels_tensor = labels_tensor
    self.colors_tensor = colors_tensor
    self._color_map = None

  def get_color_map(self, session=None):
    if self._color_map is None:
      assert self.labels_tensor is not None
      assert self.colors_tensor is not None
      session = session or tf.get_default_session()
      assert session is not None
      labels, colors = session.run(
        [self.labels_tensor, self.colors_tensor]
      )
      self._color_map = {
        k: tuple(v)
        for k, v in zip(labels, colors)
      }
    return self._color_map

  def __call__(self, inputs, session=None):
    session = session or tf.get_default_session()
    assert session is not None
    return session.run(self.outputs_tensor, feed_dict={
      self.inputs_tensor: inputs
    })

def save_inference_model(export_dir, inference_model, session=None, replace=True):
  if session is None:
    session = tf.get_default_session()
  assert session is not None
  if replace and is_directory(export_dir):
    get_logger().info('replacing %s', export_dir)
    delete_recursively(export_dir)
  prediction_signature = predict_signature_def(
    inputs={INPUTS_KEY: inference_model.inputs_tensor},
    outputs={k: v for k, v in {
        OUTPUTS_KEY: inference_model.outputs_tensor,
        LABELS_KEY: inference_model.labels_tensor,
        COLORS_KEY: inference_model.colors_tensor
      }.items() if v is not None
    }
  )
  signature_def_map = {
    DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
  }
  legacy_init_op = tf.group(
    tf.tables_initializer(),
    name='legacy_init_op'
  )
  builder = SavedModelBuilder(export_dir)
  builder.add_meta_graph_and_variables(
    session,
    [SERVING],
    signature_def_map=signature_def_map,
    legacy_init_op=legacy_init_op
  )
  builder.save()

def get_output_tensor_or_none(graph, signature, name):
  tensor_name = signature.outputs[name].name
  return graph.get_tensor_by_name(tensor_name) if tensor_name else None

def load_inference_model(export_dir, session=None):
  if session is None:
    session = tf.get_default_session()
  assert session is not None
  meta_graph_def = tf.saved_model.loader.load(session, [SERVING], export_dir)
  signature = meta_graph_def.signature_def[DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  inputs_name = signature.inputs[INPUTS_KEY].name
  outputs_name = signature.outputs[OUTPUTS_KEY].name
  get_logger().info('inputs_name: %s', inputs_name)
  get_logger().info('outputs_name: %s', outputs_name)
  graph = tf.get_default_graph()
  inputs = graph.get_tensor_by_name(inputs_name)
  outputs = graph.get_tensor_by_name(outputs_name)
  get_logger().info('inputs: %s', inputs)
  get_logger().info('output: %s', outputs)
  return InferenceModel(
    inputs,
    outputs,
    get_output_tensor_or_none(graph, signature, LABELS_KEY),
    get_output_tensor_or_none(graph, signature, COLORS_KEY)
  )
