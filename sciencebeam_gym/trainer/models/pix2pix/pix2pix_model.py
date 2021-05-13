from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import json
from functools import reduce

import tensorflow as tf


from tensorflow.python.lib.io.file_io import FileIO  # pylint: disable=E0611

from sciencebeam_gym.trainer.data.examples import (
    get_matching_files,
    read_examples
)

from sciencebeam_gym.preprocess.color_map import (
    parse_color_map_from_file
)

from sciencebeam_gym.tools.calculate_class_weights import (
    tf_calculate_efnet_weights_for_frequency_by_label
)

from sciencebeam_gym.trainer.models.pix2pix.tf_utils import (
    find_nearest_centroid_indices
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
    parse_page_range
)

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_core import (
    BaseLoss,
    ALL_BASE_LOSS,
    create_pix2pix_model,
    create_other_summaries
)

from sciencebeam_gym.trainer.models.pix2pix.evaluate import (
    evaluate_separate_channels,
    evaluate_predictions,
    evaluation_summary
)

from sciencebeam_gym.model_utils.channels import (
    calculate_color_masks
)


UNKNOWN_COLOR = (255, 255, 255)
UNKNOWN_LABEL = 'unknown'

DEFAULT_UNKNOWN_CLASS_WEIGHT = 0.1


class GraphMode(object):
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3


def get_logger():
    return logging.getLogger(__name__)


class GraphReferences(object):
    """Holder of base tensors used for training model using common task."""

    def __init__(self):
        self.is_training = None
        self.inputs = dict()
        self.examples = None
        self.train = None
        self.global_step = None
        self.metric_updates = []
        self.metric_values = []
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
        self.pos_weight = None


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
        reduce(
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
        dimension_labels_with_unknown = dimension_labels + [UNKNOWN_LABEL]
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


def parse_json_file(filename):
    with FileIO(filename, 'r') as f:
        return json.load(f)


def class_weights_to_pos_weight(
        class_weights, labels,
        use_unknown_class, unknown_class_weight=DEFAULT_UNKNOWN_CLASS_WEIGHT):

    pos_weight = [class_weights[k] for k in labels]
    return pos_weight + [unknown_class_weight] if use_unknown_class else pos_weight


def parse_color_map(color_map_filename):
    with FileIO(color_map_filename, 'r') as config_f:
        return parse_color_map_from_file(
            config_f
        )


def color_map_to_labels(color_map, labels=None):
    if labels:
        if not all(k in color_map for k in labels):
            raise ValueError(
                'not all lables found in color map, labels=%s, available keys=%s' %
                (labels, color_map.keys())
            )
        return labels
    return sorted(color_map.keys())


def color_map_to_colors(color_map, labels):
    return [color_map[k] for k in labels]


def colors_and_labels_with_unknown_class(colors, labels, use_unknown_class):
    if use_unknown_class or not colors:
        return (
            colors + [UNKNOWN_COLOR],
            labels + [UNKNOWN_LABEL]
        )
    else:
        return colors, labels


def remove_none_from_dict(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def _create_pos_weights_tensor(
        base_loss,
        separate_channel_annotation_tensor,
        pos_weight_values,
        input_uri,
        debug):

    frequency_by_label = tf.reduce_sum(
        separate_channel_annotation_tensor,
        axis=[0, 1],
        keep_dims=True,
        name='frequency_by_channel'
    )
    pos_weight_sample = tf_calculate_efnet_weights_for_frequency_by_label(
        frequency_by_label
    )
    pos_weight = (
        pos_weight_sample * pos_weight_values
        if base_loss == BaseLoss.WEIGHTED_SAMPLE_WEIGHTED_CROSS_ENTROPY
        else pos_weight_sample
    )
    if debug:
        pos_weight = tf.Print(
            pos_weight, [
                pos_weight,
                pos_weight_sample,
                frequency_by_label,
                input_uri
            ],
            'pos weights, sample, frequency, uri: ',
            summarize=1000
        )
    get_logger().debug(
        'pos_weight before batch: %s (frequency_by_label: %s)',
        pos_weight, frequency_by_label
    )
    return pos_weight


class Model(object):
    def __init__(self, args):
        self.args = args
        self.image_width = 256
        self.image_height = 256
        self.color_map = None
        self.pos_weight = None
        self.dimension_colors = None
        self.dimension_labels = None
        self.use_unknown_class = args.use_unknown_class
        self.use_separate_channels = args.use_separate_channels and self.args.color_map is not None
        logger = get_logger()
        logger.info('use_separate_channels: %s', self.use_separate_channels)
        if self.args.color_map:
            color_map = parse_color_map(args.color_map)
            class_weights = (
                parse_json_file(self.args.class_weights)
                if (
                    self.args.class_weights and
                    self.args.base_loss in {
                        BaseLoss.WEIGHTED_CROSS_ENTROPY,
                        BaseLoss.WEIGHTED_SAMPLE_WEIGHTED_CROSS_ENTROPY
                    }
                )
                else None
            )
            available_labels = color_map_to_labels(color_map)
            if class_weights:
                # remove labels with zero class weights
                available_labels = [k for k in available_labels if class_weights.get(k, 0.0) != 0.0]
            self.dimension_labels = args.channels if args.channels else available_labels
            self.dimension_colors = color_map_to_colors(color_map, self.dimension_labels)
            self.dimension_colors_with_unknown, self.dimension_labels_with_unknown = (
                colors_and_labels_with_unknown_class(
                    self.dimension_colors,
                    self.dimension_labels,
                    self.use_unknown_class
                )
            )
            logger.debug("dimension_colors: %s", self.dimension_colors)
            logger.debug("dimension_labels: %s", self.dimension_labels)
            if class_weights:
                self.pos_weight = class_weights_to_pos_weight(
                    class_weights,
                    self.dimension_labels,
                    self.use_separate_channels,
                    class_weights.get(UNKNOWN_LABEL, DEFAULT_UNKNOWN_CLASS_WEIGHT)
                )
                logger.info("pos_weight: %s", self.pos_weight)

    def _build_predict_graph(self):
        tensors = GraphReferences()
        input_image_tensor = tf.placeholder(
            tf.uint8, (None, None, None, 3),
            name='inputs_image'
        )
        tensors.inputs = dict(
            image=input_image_tensor
        )

        tensors.image_tensor = tf.image.resize_images(
            tf.image.convert_image_dtype(input_image_tensor, tf.float32),
            (self.image_height, self.image_width),
            method=tf.image.ResizeMethod.BILINEAR
        )

        if self.use_separate_channels:
            n_output_channels = len(self.dimension_labels_with_unknown)
        else:
            n_output_channels = 3
        pix2pix_model = create_pix2pix_model(
            tensors.image_tensor,
            None,
            self.args,
            is_training=False,
            pos_weight=tensors.pos_weight,
            n_output_channels=n_output_channels
        )
        tensors.pred = pix2pix_model.outputs
        return tensors

    def build_graph(self, data_paths, batch_size, graph_mode):
        if graph_mode == GraphMode.PREDICT:
            return self._build_predict_graph()

        logger = get_logger()
        logger.debug('batch_size: %s', batch_size)
        tensors = GraphReferences()
        tensors.is_training = tf.constant(graph_mode == GraphMode.TRAIN)
        is_training = (
            graph_mode == GraphMode.TRAIN or
            graph_mode == GraphMode.EVALUATE
        )

        if not data_paths:
            raise ValueError('data_paths required')
        get_logger().info('reading examples from %s', data_paths)
        tensors.examples = read_examples(
            get_matching_files(data_paths),
            shuffle=(graph_mode == GraphMode.TRAIN),
            num_epochs=None if is_training else 2,
            page_range=self.args.pages,
            channel_colors=(
                self.dimension_colors if self.args.filter_annotated
                else None
            )
        )
        parsed = tensors.examples

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
            with tf.variable_scope('channels'):
                color_masks = calculate_color_masks(
                    tensors.annotation_tensor,
                    self.dimension_colors,
                    use_unknown_class=self.use_unknown_class
                )
                tensors.separate_channel_annotation_tensor = tf.stack(color_masks, axis=-1)
                if self.args.base_loss == BaseLoss.SAMPLE_WEIGHTED_CROSS_ENTROPY:
                    with tf.variable_scope('class_weights'):
                        tensors.pos_weight = _create_pos_weights_tensor(
                            base_loss=self.args.base_loss,
                            separate_channel_annotation_tensor=(
                                tensors.separate_channel_annotation_tensor
                            ),
                            pos_weight_values=self.pos_weight,
                            input_uri=tensors.input_uri,
                            debug=self.args.debug
                        )
        else:
            tensors.annotation_tensor = tf.image.convert_image_dtype(
                tensors.annotation_tensor, tf.float32
            )
            tensors.separate_channel_annotation_tensor = tensors.annotation_tensor

        batched_tensors: dict = tf.train.batch(
            remove_none_from_dict({
                k: getattr(tensors, k)
                for k in {
                    'input_uri',
                    'annotation_uri',
                    'image_tensor',
                    'annotation_tensor',
                    'separate_channel_annotation_tensor',
                    'pos_weight'
                }
            }),
            batch_size=batch_size
        )
        for k, v in batched_tensors.items():
            setattr(tensors, k, v)

        if tensors.pos_weight is None:
            tensors.pos_weight = self.pos_weight

        pix2pix_model = create_pix2pix_model(
            tensors.image_tensor,
            tensors.separate_channel_annotation_tensor,
            self.args,
            is_training=tensors.is_training,
            pos_weight=tensors.pos_weight
        )

        if self.use_separate_channels:
            with tf.name_scope("evaluation"):
                tensors.output_layer_labels = tf.constant(self.dimension_labels_with_unknown)
                evaluation_result = evaluate_separate_channels(
                    targets=pix2pix_model.targets,
                    outputs=pix2pix_model.outputs
                )
                tensors.evaluation_result = evaluation_result
                evaluation_summary(evaluation_result, self.dimension_labels_with_unknown)
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
                        n_classes=len(self.dimension_colors_with_unknown)
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

        if (
            self.args.base_loss == BaseLoss.SAMPLE_WEIGHTED_CROSS_ENTROPY and
            tensors.pos_weight is not None
        ):
            with tf.variable_scope('pos_weight_summary'):
                tf.summary.text('pos_weight', tf.as_string(tf.reshape(
                    tensors.pos_weight, [-1, int(tensors.pos_weight.shape[-1])]
                )))

        tensors.summary = tf.summary.merge_all()
        return tensors

    def build_train_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMode.TRAIN)

    def build_eval_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMode.EVALUATE)

    def build_predict_graph(self):
        return self.build_graph(None, None, GraphMode.PREDICT)

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


def str_to_list(s):
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(',')]


def model_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ngf", type=int, default=64, help="number of generator filters in first conv layer"
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="number of discriminator filters in first conv layer"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="initial learning rate for adam"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="momentum term of adam"
    )
    parser.add_argument(
        "--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient"
    )
    parser.add_argument(
        "--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient"
    )

    parser.add_argument(
        '--pages', type=parse_page_range, default=None,
        help='only processes the selected pages'
    )
    parser.add_argument(
        '--color_map',
        type=str,
        help='The path to the color map configuration.'
    )
    parser.add_argument(
        '--class_weights',
        type=str,
        help='The path to the class weights configuration.'
    )
    parser.add_argument(
        '--channels',
        type=str_to_list,
        help='The channels to use (subset of color map), otherwise all of the labels will be used'
    )
    parser.add_argument(
        '--filter_annotated',
        type=str_to_bool,
        default=False,
        help='Only include pages that have annotations for the selected channels'
        ' (if color map is provided)'
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
        '--use_separate_discriminator_channels',
        type=str_to_bool,
        default=False,
        help='The separate discriminator channels per annotation (if color map is provided)'
    )
    parser.add_argument(
        '--use_separate_discriminators',
        type=str_to_bool,
        default=False,
        help='The separate discriminators per annotation (if color map is provided)'
    )
    parser.add_argument(
        '--base_loss',
        type=str,
        default=BaseLoss.L1,
        choices=ALL_BASE_LOSS,
        help='The base loss function to use'
    )
    parser.add_argument(
        '--debug',
        type=str_to_bool,
        default=True,
        help='Enable debug mode'
    )
    return parser


def create_model(argv=None):
    """Factory method that creates model to be used by generic task.py."""
    parser = model_args_parser()
    args, task_args = parser.parse_known_args(argv)
    return Model(args), task_args
