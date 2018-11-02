import logging

import pytest
import tensorflow as tf

from sciencebeam_utils.utils.collection import (
    to_namedtuple
)

from sciencebeam_gym.utils.tfrecord import (
    dict_to_example
)

from sciencebeam_gym.trainer.data.examples import (
    parse_example,
    MapKeysTracker
)

from sciencebeam_gym.trainer.data.examples_test import (
    EXAMPLE_PROPS_1,
    list_dataset
)

from sciencebeam_gym.trainer.evaluator import (
    Evaluator
)

TEST_PATH = '.temp/test/evaluator'

CHECKPOINT_PATH = TEST_PATH + '/checkpoints'
DATA_PATHS = [TEST_PATH + '/preproc']
OUTPUT_PATH = TEST_PATH + '/output'
BATCH_SIZE = 10
EVAL_SET_SIZE = 10

DEFAULT_ARGS = dict(
    batch_size=BATCH_SIZE,
    eval_set_size=EVAL_SET_SIZE,
    streaming_eval=False,
    output_path=OUTPUT_PATH
)

DEFAULT_KWARGS = dict(
    checkpoint_path=CHECKPOINT_PATH,
    data_paths=DATA_PATHS,
    eval_batch_size=BATCH_SIZE
)


def get_logger():
    return logging.getLogger(__name__)


def setup_module():
    logging.basicConfig(level='DEBUG')


class GraphMode(object):
    TRAIN = 'train'
    EVALUATE = 'eval'


def example_dataset(map_keys_tracker, examples):
    dataset = list_dataset([
        dict_to_example(example).SerializeToString()
        for example in examples
    ], tf.string)
    dataset = dataset.map(map_keys_tracker.wrap(parse_example))
    return dataset


class ExampleModel(object):
    def __init__(self, examples):
        self.examples = examples

    def build_graph(self, data_paths, batch_size, graph_mode):  # pylint: disable=unused-argument
        tensors = dict()
        tensors['is_training'] = tf.placeholder(tf.bool)
        map_keys_tracker = MapKeysTracker()
        dataset = example_dataset(map_keys_tracker, self.examples)
        iterator = dataset.make_one_shot_iterator()
        parsed = map_keys_tracker.unwrap(iterator.get_next())
        get_logger().debug('parsed: %s', parsed)
        tensors['examples'] = parsed
        tensors['metric_values'] = []
        tensors['metric_updates'] = []
        tensors['global_step'] = tf.constant(100, tf.int32)
        tensors['summaries'] = dict()
        tensors['image_tensors'] = dict()
        tensors['evaluation_result'] = None
        image_shape = (10, 10, 3)
        pre_batch_tensors = {
            'input_uri': tf.squeeze(parsed['input_uri']),
            'annotation_uri': tf.squeeze(parsed['annotation_uri']),
            'image_tensor': tf.zeros(image_shape),
            'annotation_tensor': tf.zeros(image_shape),
            'output_image': tf.zeros(image_shape)
        }
        post_batch_tensors = tf.train.batch(pre_batch_tensors, batch_size=batch_size)
        tensors.update(post_batch_tensors)
        for name in ['image_tensor', 'annotation_tensor', 'output_image']:
            image_tensor = tensors[name]
            get_logger().debug('name=%s, image_tensor=%s', name, image_tensor)
            tf.summary.image(name, image_tensor)
            tensors['image_tensors'][name] = image_tensor
            tensors['summaries'][name] = image_tensor
        tensors['summary'] = tf.summary.merge_all()
        tensors['initializer'] = [tf.global_variables_initializer()]
        return to_namedtuple(tensors, name='Tensors')

    def build_train_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMode.TRAIN)

    def build_eval_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMode.EVALUATE)


@pytest.mark.slow
class TestEvaluator(object):
    def test_should_not_fail_eval_in_session(self):
        with tf.Graph().as_default():
            model = ExampleModel([EXAMPLE_PROPS_1] * BATCH_SIZE)
            tensors = model.build_train_graph(
                DATA_PATHS, BATCH_SIZE
            )

            evaluator = Evaluator(
                args=to_namedtuple(DEFAULT_ARGS, name='args'),
                model=model,
                **DEFAULT_KWARGS
            )
            evaluator.init()

            get_logger().info('starting session')
            with tf.Session() as session:
                coord = tf.train.Coordinator()
                tf.train.start_queue_runners(sess=session, coord=coord)
                get_logger().info('evaluating')
                session.run(tensors.initializer)
                evaluator.evaluate_in_session(session, tensors)
            get_logger().info('done')
