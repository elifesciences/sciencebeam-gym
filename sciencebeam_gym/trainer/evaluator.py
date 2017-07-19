import os
import logging
import json
from io import BytesIO

import matplotlib as mpl
# this is important to run on the cloud - we won't have python-tk installed
mpl.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import six

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from PIL import Image

from sciencebeam_gym.trainer.util import (
  CustomSupervisor,
  FileIO
)

def get_logger():
  return logging.getLogger(__name__)

def plot_image(ax, image, label):
  if len(image.shape) == 3:
    get_logger().info('image shape: %s (%s)', image.shape, image.shape[-1])
    if image.shape[-1] == 1:
      ax.imshow(image.squeeze(), aspect='auto', vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
    else:
      ax.imshow(image, aspect='auto')
  else:
    ax.imshow(np.dstack((image.astype(np.uint8),)*3)*100, aspect='auto')
  ax.set_title(label)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)

def show_result_images3(title, input_image, annot, output_image):
  fig, (ax_img, ax_annot, ax_out) = plt.subplots(1, 3, sharey=True)

  plt.suptitle(title)

  plot_image(ax_img, input_image, 'Input')
  plot_image(ax_annot, annot, 'Label')
  plot_image(ax_out, output_image, 'Output')
  return fig

def save_image_data(image_filename, image_data):
  image = Image.fromarray(np.array(image_data), "RGB")
  with FileIO(image_filename, 'wb') as image_f:
    image.save(image_f, 'png')

def save_file(filename, data):
  with FileIO(filename, 'wb') as f:
    f.write(data)


IMAGE_PREFIX = 'image_'

class Evaluator(object):
  """Loads variables from latest checkpoint and performs model evaluation."""

  def __init__(
    self, args, model,
    checkpoint_path, data_paths, dataset='eval',
    run_async=None):

    self.eval_batch_size = args.eval_batch_size
    self.num_eval_batches = args.eval_set_size // self.eval_batch_size
    self.batch_of_examples = []
    self.checkpoint_path = checkpoint_path
    self.output_path = os.path.join(args.output_path, dataset)
    self.eval_data_paths = data_paths
    self.batch_size = args.batch_size
    self.stream = args.streaming_eval
    self.model = model
    self.results_dir = os.path.join(self.output_path, 'results')
    self.run_async = run_async
    if not run_async:
      self.run_async = lambda f, args: f(*args)

  def init(self):
    file_io.recursive_create_dir(self.results_dir)

  def _check_fetches(self, fetches):
    for k, v in six.iteritems(fetches):
      if v is None:
        raise Exception('fetches tensor is None: {}'.format(k))

  def _get_default_fetches(self, tensors):
    return {
      'global_step': tensors.global_step,
      'input_uri': tensors.input_uri,
      'input_image': tensors.image_tensor,
      'annotation_image': tensors.annotation_tensor,
      'output_image': tensors.summaries.get('output_image'),
      'metric_values': tensors.metric_values
    }

  def _add_image_fetches(self, fetches, tensors):
    for k, v in six.iteritems(tensors.image_tensors):
      fetches[IMAGE_PREFIX + k] = v
    return fetches

  def _save_prediction_summary_image(self, eval_index, results):
    logger = get_logger()
    global_step = results['global_step']
    metric_values = results['metric_values']
    for batch_index, input_uri in enumerate(results['input_uri']):
      pred_image = results['input_image'][batch_index]
      pred_annotation = results['annotation_image'][batch_index]
      pred_np = results['output_image'][batch_index]

      logger.info('input_uri: %s', input_uri)
      fig = show_result_images3(
        'Iteration {} (...{})'.format(
          global_step,
          input_uri[-50:]
        ),
        pred_image,
        pred_annotation,
        pred_np
      )
      result_file = os.path.join(
        self.results_dir, 'result_{}_{}_{}_summary_{}.png'.format(
          global_step, eval_index, batch_index, metric_values[0]
        )
      )
      logging.info('result_file: %s', result_file)
      bio = BytesIO()
      plt.savefig(bio, format='png')
      plt.close(fig)
      self.run_async(save_file, (result_file, bio.getvalue()))

  def _save_result_images(self, eval_index, results):
    global_step = results['global_step']
    for k in six.iterkeys(results):
      if k.startswith(IMAGE_PREFIX):
        batch_image_data = results[k]
        name = k[len(IMAGE_PREFIX):]
        for batch_index, image_data in enumerate(batch_image_data):
          image_filename = os.path.join(
            self.results_dir, 'result_{}_{}_{}_{}.png'.format(
              global_step, eval_index, batch_index, name
            )
          )
          self.run_async(save_image_data, (image_filename, image_data))

  def _save_meta(self, eval_index, results):
    global_step = results['global_step']
    metric_values = results['metric_values']
    for batch_index, input_uri in enumerate(results['input_uri']):
      meta_file = os.path.join(
        self.results_dir, 'result_{}_{}_{}_meta.json'.format(
          global_step, eval_index, batch_index
        )
      )
      meta_str = json.dumps({
        'global_step': global_step,
        'eval_index': eval_index,
        'batch_index': batch_index,
        'metric_values': [float(x) for x in metric_values],
        'input_uri': input_uri
      }, indent=2)
      with FileIO(meta_file, 'w') as meta_f:
        meta_f.write(meta_str)

  def evaluate_in_session(self, session, tensors, num_eval_batches=None):
    summary_writer = tf.summary.FileWriter(self.output_path)
    num_eval_batches = num_eval_batches or self.num_eval_batches
    if self.stream:
      for _ in range(num_eval_batches):
        session.run(tensors.metric_updates)
    else:
      if not self.batch_of_examples:
        for _ in range(num_eval_batches):
          self.batch_of_examples.append(session.run(tensors.examples))

      for eval_index in range(num_eval_batches):
        session.run(tensors.metric_updates, {
          tensors.examples: self.batch_of_examples[eval_index]
        })

        fetches = self._get_default_fetches(tensors)
        self._add_image_fetches(fetches, tensors)
        fetches['summary_value'] = tensors.summary
        self._check_fetches(fetches)
        results = session.run(fetches)

        self._save_prediction_summary_image(eval_index, results)
        self._save_result_images(eval_index, results)
        self._save_meta(eval_index, results)

        global_step = results['global_step']
        summary_value = results['summary_value']
        summary_writer.add_summary(summary_value, global_step)
        summary_writer.flush()

        metric_values = results['metric_values']

        logging.info('eval done')
        return metric_values

    metric_values = session.run(tensors.metric_values)
    return metric_values

  def evaluate(self, num_eval_batches=None, session=None):
    """Run one round of evaluation, return loss and accuracy."""

    num_eval_batches = num_eval_batches or self.num_eval_batches
    with tf.Graph().as_default() as graph:
      tensors = self.model.build_eval_graph(
        self.eval_data_paths,
        self.eval_batch_size
      )

      saver = tf.train.Saver()

    sv = CustomSupervisor(
      model=self.model,
      graph=graph,
      logdir=self.output_path,
      summary_op=None,
      global_step=None,
      saver=saver
    )
    try:
      last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
      logging.info('last_checkpoint: %s (%s)', last_checkpoint, self.checkpoint_path)

      file_io.recursive_create_dir(self.results_dir)

      with sv.managed_session(
          master='', start_standard_services=False) as session:
        sv.saver.restore(session, last_checkpoint)

        logging.info('session restored')

        if self.stream:
          logging.info('start queue runners (stream)')
          sv.start_queue_runners(session)
          for _ in range(num_eval_batches):
            session.run(self.tensors.metric_updates)
        else:
          if not self.batch_of_examples:
            logging.info('start queue runners (batch)')
            sv.start_queue_runners(session)
            for i in range(num_eval_batches):
              self.batch_of_examples.append(session.run(tensors.examples))
          else:
            logging.info('starting queue runners, has batch of examples')
            sv.start_queue_runners(session)

          logging.info('updating metrics')
          for i in range(num_eval_batches):
            session.run(tensors.metric_updates,
                        {tensors.examples: self.batch_of_examples[i]})

        logging.info('evaluate_in_session')
        return self.evaluate_in_session(session, tensors)
    finally:
      sv.stop()

  def write_predictions(self):
    """Run one round of predictions and write predictions to csv file."""
    num_eval_batches = min(10, self.num_eval_batches + 1)
    with tf.Graph().as_default() as graph:
      tensors = self.model.build_eval_graph(
        self.eval_data_paths,
        self.batch_size
      )
      saver = tf.train.Saver()

    sv = CustomSupervisor(
      model=self.model,
      graph=graph,
      logdir=self.output_path,
      summary_op=None,
      global_step=None,
      saver=saver
    )

    file_io.recursive_create_dir(self.results_dir)

    last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_path)
    with sv.managed_session(
        master='', start_standard_services=False) as session:
      sv.saver.restore(session, last_checkpoint)
      predictions_filename = os.path.join(self.output_path, 'predictions.csv')
      with FileIO(predictions_filename, 'w') as csv_f:
        sv.start_queue_runners(session)
        last_log_progress = 0
        for eval_index in range(num_eval_batches):
          progress = eval_index * 100 // num_eval_batches
          if progress > last_log_progress:
            logging.info('%3d%% predictions processed', progress)
            last_log_progress = progress

          fetches = self._get_default_fetches(tensors)
          self._add_image_fetches(fetches, tensors)
          self._check_fetches(fetches)
          results = session.run(fetches)

          self._save_prediction_summary_image(eval_index, results)
          self._save_result_images(eval_index, results)
          self._save_meta(eval_index, results)

          input_uri = results['input_uri']
          metric_values = results['metric_values']
          csv_f.write('{},{}\n'.format(input_uri, metric_values[0]))
