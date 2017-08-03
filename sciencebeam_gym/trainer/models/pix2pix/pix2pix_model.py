from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse

import tensorflow as tf

import six
from six.moves.configparser import ConfigParser

from tensorflow.python.lib.io.file_io import FileIO

from sciencebeam_gym.trainer.util import (
  read_examples
)

from sciencebeam_gym.tools.colorize_image import (
  parse_color_map_from_configparser
)

from sciencebeam_gym.trainer.models.pix2pix.tf_utils import (
  find_nearest_centroid_indices
)

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_core import (
  BaseLoss,
  create_pix2pix_model,
  create_other_summaries
)

from sciencebeam_gym.trainer.models.pix2pix.evaluate import (
  evaluate_separate_channels,
  evaluate_predictions,
  evaluation_summary
)


class GraphMode(object):
  TRAIN = 1
  EVALUATE = 2
  PREDICT = 3

def get_logger():
  return logging.getLogger(__name__)


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []
    self.input_jpeg = None
    self.input_uri = None
    self.image_tensor = None
    self.annotation_uri = None
    self.annotation_tensor = None
    self.separate_channel_annotation_tensor = None
    self.class_labels_tensor = None
    self.pred = None
    self.probabilities = None
    self.summary = None
    self.summaries = None
    self.image_tensors = None
    self.targets_class_indices = None
    self.outputs_class_indices = None
    self.output_layer_labels = None
    self.evaluation_result = None

def colors_to_dimensions(image_tensor, colors, use_unknown_class=False):
  with tf.variable_scope("colors_to_dimensions"):
    single_label_tensors = []
    ones = tf.fill(image_tensor.shape[0:-1], 1.0, name='ones')
    zeros = tf.fill(ones.shape, 0.0, name='zeros')
    for single_label_color in colors:
      i = len(single_label_tensors)
      with tf.variable_scope("channel_{}".format(i)):
        is_color = tf.reduce_all(
          tf.equal(image_tensor, single_label_color),
          axis=-1,
          name='is_color'
        )
        single_label_tensor = tf.where(
          is_color,
          ones,
          zeros
        )
        single_label_tensors.append(single_label_tensor)
    if use_unknown_class:
      with tf.variable_scope("unknown_class"):
        single_label_tensors.append(
          tf.where(
            tf.add_n(single_label_tensors) < 0.5,
            ones,
            zeros
          )
        )
    return tf.stack(single_label_tensors, axis=-1)

def batch_dimensions_to_colors_list(image_tensor, colors):
  batch_images = []
  for i, single_label_color in enumerate(colors):
    batch_images.append(
      tf.expand_dims(
        image_tensor[:, :, :, i],
        axis=-1
      ) * ([x / 255.0 for x in single_label_color])
    )
  return batch_images

def batch_dimensions_to_most_likely_colors_list(image_tensor, colors):
  with tf.variable_scope("batch_dimensions_to_most_likely_colors_list"):
    colors_tensor = tf.constant(colors, dtype=tf.uint8, name='colors')
    most_likely_class_index = tf.argmax(image_tensor, 3)
    return tf.gather(params=colors_tensor, indices=most_likely_class_index)

def add_summary_image(tensors, name, image):
  tensors.image_tensors[name] = image
  tf.summary.image(name, image)

def convert_image(image_tensor):
  return tf.image.convert_image_dtype(
    image_tensor,
    dtype=tf.uint8,
    saturate=True
  )

def add_simple_summary_image(tensors, name, image_tensor):
  with tf.name_scope(name):
    add_summary_image(
      tensors,
      name,
      convert_image(image_tensor)
    )

def replace_black_with_white_color(image_tensor):
  is_black = tf.reduce_all(
  tf.equal(image_tensor, (0, 0, 0)),
    axis=-1
  )
  is_black = tf.stack([is_black] * 3, axis=-1)
  return tf.where(
    is_black,
    255 * tf.ones_like(image_tensor),
    image_tensor
  )

def combine_image(batch_images, replace_black_with_white=False):
  clipped_batch_images = [
    tf.clip_by_value(batch_image, 0.0, 1.0)
    for batch_image in batch_images
  ]
  combined_image = convert_image(
    six.moves.reduce(
      lambda a, b: a + b,
      clipped_batch_images
    )
  )
  if replace_black_with_white:
    combined_image = replace_black_with_white_color(combined_image)
  return combined_image

def remove_last(a):
  return a[:-1]

def add_model_summary_images(
  tensors, dimension_colors, dimension_labels,
  use_separate_channels=False,
  has_unknown_class=False):

  tensors.summaries = {}
  add_simple_summary_image(
    tensors, 'input', tensors.image_tensor
  )
  add_simple_summary_image(
    tensors, 'target', tensors.annotation_tensor
  )
  if (has_unknown_class or not use_separate_channels) and dimension_labels is not None:
    dimension_labels_with_unknown = dimension_labels + ['unknown']
    dimension_colors_with_unknown = dimension_colors + [(255, 255, 255)]
  else:
    dimension_labels_with_unknown = dimension_labels
    dimension_colors_with_unknown = dimension_colors
  if use_separate_channels:
    for name, outputs in [
        ('targets', tensors.separate_channel_annotation_tensor),
        ('outputs', tensors.pred)
      ]:

      batch_images = batch_dimensions_to_colors_list(
        outputs,
        dimension_colors_with_unknown
      )
      batch_images_excluding_unknown = (
        remove_last(batch_images)
        if has_unknown_class
        else batch_images
      )
      for i, (batch_image, dimension_label) in enumerate(zip(
        batch_images, dimension_labels_with_unknown)):

        suffix = "_{}_{}".format(
          i, dimension_label if dimension_label else 'unknown_label'
        )
        add_simple_summary_image(
          tensors, name + suffix, batch_image
        )
      with tf.name_scope(name + "_combined"):
        combined_image = combine_image(batch_images_excluding_unknown)
        if name == 'outputs':
          tensors.summaries['output_image'] = combined_image
        add_summary_image(
          tensors,
          name + "_combined",
          combined_image
        )

      if name == 'outputs':
        with tf.name_scope(name + "_most_likely"):
          add_summary_image(
            tensors,
            name + "_most_likely",
            batch_dimensions_to_most_likely_colors_list(
              outputs,
              dimension_colors_with_unknown)
          )
  else:
    add_simple_summary_image(
      tensors,
      "output",
      tensors.pred
    )
    if tensors.outputs_class_indices is not None:
      outputs = tensors.pred
      with tf.name_scope("outputs_most_likely"):
        colors_tensor = tf.constant(
          dimension_colors_with_unknown,
          dtype=tf.uint8, name='colors'
        )
        add_summary_image(
          tensors,
          "outputs_most_likely",
          tf.gather(
            params=colors_tensor,
            indices=tensors.outputs_class_indices
          )
        )
    tensors.summaries['output_image'] = tensors.image_tensors['output']

class Model(object):
  def __init__(self, args):
    self.args = args
    self.image_width = 256
    self.image_height = 256
    self.color_map = None
    self.dimension_colors = None
    self.dimension_labels = None
    self.use_unknown_class = args.use_unknown_class
    self.use_separate_channels = args.use_separate_channels and self.args.color_map is not None
    logger = get_logger()
    logger.info('use_separate_channels: %s', self.use_separate_channels)
    if self.args.color_map:
      color_map_config = ConfigParser()
      with FileIO(self.args.color_map, 'r') as config_f:
        color_map_config.readfp(config_f)
      self.color_map = parse_color_map_from_configparser(color_map_config)
      color_label_map = {
        (int(k), int(k), int(k)): v
        for k, v in color_map_config.items('color_labels')
      }
      sorted_keys = sorted(six.iterkeys(self.color_map))
      self.dimension_colors = [self.color_map[k] for k in sorted_keys]
      self.dimension_labels = [color_label_map.get(k) for k in sorted_keys]
      logger.debug("dimension_colors: %s", self.dimension_colors)
      logger.debug("dimension_labels: %s", self.dimension_labels)
      if self.use_unknown_class or not self.dimension_colors:
        self.dimension_labels_with_unknown = self.dimension_labels + ['unknown']
        self.dimension_colors_with_unknown = self.dimension_colors + [(255, 255, 255)]
      else:
        self.dimension_labels_with_unknown = self.dimension_labels
        self.dimension_colors_with_unknown = self.dimension_colors

  def build_graph(self, data_paths, batch_size, graph_mode):
    logger = get_logger()
    logger.debug('batch_size: %s', batch_size)
    tensors = GraphReferences()
    is_training = (
      graph_mode == GraphMode.TRAIN or
      graph_mode == GraphMode.EVALUATE
    )
    if data_paths:
      tensors.keys, tensors.examples = read_examples(
        data_paths,
        shuffle=(graph_mode == GraphMode.TRAIN),
        num_epochs=None if is_training else 2
      )
    else:
      tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))
    with tf.name_scope('inputs'):
      feature_map = {
        'input_uri':
          tf.FixedLenFeature(
            shape=[], dtype=tf.string, default_value=['']
          ),
        'annotation_uri':
          tf.FixedLenFeature(
            shape=[], dtype=tf.string, default_value=['']
          ),
        'input_image':
          tf.FixedLenFeature(
            shape=[], dtype=tf.string
          ),
        'annotation_image':
          tf.FixedLenFeature(
            shape=[], dtype=tf.string
          )
      }
      logging.info('tensors.examples: %s', tensors.examples)
    parsed = tf.parse_single_example(tensors.examples, features=feature_map)

    tensors.image_tensors = {}

    tensors.input_uri = tf.squeeze(parsed['input_uri'])
    tensors.annotation_uri = tf.squeeze(parsed['annotation_uri'])
    raw_input_image = tf.squeeze(parsed['input_image'])
    logging.info('raw_input_image: %s', raw_input_image)
    raw_annotation_image = tf.squeeze(parsed['annotation_image'])
    tensors.image_tensor = tf.image.decode_png(raw_input_image, channels=3)
    tensors.annotation_tensor = tf.image.decode_png(raw_annotation_image, channels=3)

    # TODO resize_images and tf.cast did not work on input image
    #   but did work on annotation image
    tensors.image_tensor = tf.image.resize_image_with_crop_or_pad(
      tensors.image_tensor, self.image_height, self.image_width
    )

    tensors.image_tensor = tf.image.convert_image_dtype(tensors.image_tensor, tf.float32)

    tensors.annotation_tensor = tf.image.resize_image_with_crop_or_pad(
      tensors.annotation_tensor, self.image_height, self.image_width
    )

    if self.use_separate_channels:
      tensors.separate_channel_annotation_tensor = colors_to_dimensions(
        tensors.annotation_tensor,
        self.dimension_colors,
        use_unknown_class=self.use_unknown_class
      )
    else:
      tensors.annotation_tensor = tf.image.convert_image_dtype(tensors.annotation_tensor, tf.float32)
      tensors.separate_channel_annotation_tensor = tensors.annotation_tensor

    (
      tensors.input_uri,
      tensors.annotation_uri,
      tensors.image_tensor,
      tensors.annotation_tensor,
      tensors.separate_channel_annotation_tensor
    ) = tf.train.batch(
      [
        tensors.input_uri,
        tensors.annotation_uri,
        tensors.image_tensor,
        tensors.annotation_tensor,
        tensors.separate_channel_annotation_tensor
      ],
      batch_size=batch_size
    )

    pix2pix_model = create_pix2pix_model(
      tensors.image_tensor,
      tensors.separate_channel_annotation_tensor,
      self.args
    )

    if self.use_separate_channels:
      with tf.name_scope("evaluation"):
        tensors.output_layer_labels = tf.constant(self.dimension_labels)
        evaluation_result = evaluate_separate_channels(
          targets=pix2pix_model.targets,
          outputs=pix2pix_model.outputs,
          has_unknown_class=self.use_unknown_class
        )
        tensors.evaluation_result = evaluation_result
        evaluation_summary(evaluation_result, self.dimension_labels)
    else:
      with tf.name_scope('evaluation'):
        if self.dimension_colors:
          tensors.output_layer_labels = tf.constant(self.dimension_labels)
          colors_tensor = tf.constant(
            self.dimension_colors_with_unknown,
            dtype=tf.float32
          ) / 255.0
          tensors.outputs_class_indices = find_nearest_centroid_indices(
            predictions=pix2pix_model.outputs,
            centroids=colors_tensor
          )
          tensors.targets_class_indices = find_nearest_centroid_indices(
            predictions=pix2pix_model.targets,
            centroids=colors_tensor
          )
          evaluation_result = evaluate_predictions(
            labels=tensors.targets_class_indices,
            predictions=tensors.outputs_class_indices,
            n_classes=len(self.dimension_colors_with_unknown),
            has_unknown_class=self.use_unknown_class
          )
          tensors.evaluation_result = evaluation_result
          evaluation_summary(evaluation_result, self.dimension_labels)

    tensors.global_step = pix2pix_model.global_step
    tensors.train = pix2pix_model.train
    tensors.class_labels_tensor = tensors.annotation_tensor
    tensors.pred = pix2pix_model.outputs
    tensors.probabilities = pix2pix_model.outputs
    tensors.metric_values = [pix2pix_model.discrim_loss]

    add_model_summary_images(
      tensors,
      self.dimension_colors,
      self.dimension_labels,
      use_separate_channels=self.use_separate_channels,
      has_unknown_class=self.use_unknown_class
    )

    # tensors.summaries = create_summaries(pix2pix_model)
    create_other_summaries(pix2pix_model)

    tensors.summary = tf.summary.merge_all()
    return tensors

  def build_train_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMode.TRAIN)

  def build_eval_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMode.EVALUATE)

  def initialize(self, session):
    pass

  def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    accuracy_str = 'N/A'
    try:
      loss_str = '%.3f' % metric_values[0]
      accuracy_str = '%.3f' % metric_values[1]
    except (TypeError, IndexError):
      pass

    return '%s, %s' % (loss_str, accuracy_str)

def str_to_bool(s):
  return s.lower() in ('yes', 'true', '1')

def model_args_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
  parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
  parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
  parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
  parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
  parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

  parser.add_argument(
    '--color_map',
    type=str,
    help='The path to the color map configuration.'
  )
  parser.add_argument(
    '--use_unknown_class',
    type=str_to_bool,
    default=True,
    help='Use unknown class channel (if color map is provided)'
  )
  parser.add_argument(
    '--use_separate_channels',
    type=str_to_bool,
    default=False,
    help='The separate output channels per annotation (if color map is provided)'
  )
  parser.add_argument(
    '--base_loss',
    type=str,
    default=BaseLoss.L1,
    choices=[BaseLoss.L1, BaseLoss.CROSS_ENTROPY],
    help='The base loss function to use'
  )
  return parser


def create_model(argv=None):
  """Factory method that creates model to be used by generic task.py."""
  parser = model_args_parser()
  args, task_args = parser.parse_known_args(argv)
  return Model(args), task_args
