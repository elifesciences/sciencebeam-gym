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

def precision_from_tp_fp(tp, fp):
  return tp / (tp + fp)

def recall_from_tp_fn(tp, fn):
  return tp / (tp + fn)

def f1_from_precision_recall(precision, recall):
  return 2 * precision * recall / (precision + recall)

def f1_from_tp_fp_fn(tp, fp, fn):
  return f1_from_precision_recall(
    precision_from_tp_fp(tp, fp),
    recall_from_tp_fn(tp, fn)
  )

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

  def _add_evaluation_result_fetches(self, fetches, tensors):
    if tensors.evaluation_result:
      fetches['output_layer_labels'] = tensors.output_layer_labels
      fetches['confusion_matrix'] = tensors.evaluation_result.confusion_matrix
      fetches['tp'] = tensors.evaluation_result.tp
      fetches['fp'] = tensors.evaluation_result.fp
      fetches['fn'] = tensors.evaluation_result.fn
      fetches['tn'] = tensors.evaluation_result.tn
      fetches['accuracy'] = tensors.evaluation_result.accuracy
      fetches['micro_f1'] = tensors.evaluation_result.micro_f1
    return fetches

  def _accumulate_evaluation_results(self, results, accumulated_results=None):
    if accumulated_results is None:
      accumulated_results = []
    accumulated_results.append({
      'output_layer_labels': results['output_layer_labels'],
      'confusion_matrix': results['confusion_matrix'],
      'tp': results['tp'],
      'fp': results['fp'],
      'fn': results['fn'],
      'tn': results['tn'],
      'accuracy': results['accuracy'],
      'micro_f1': results['micro_f1'],
      'count': self.batch_size,
      'global_step': results['global_step']
    })
    return accumulated_results

  def _save_accumulate_evaluation_results(self, accumulated_results):
    if accumulated_results:
      global_step = accumulated_results[0]['global_step']
      output_layer_labels = accumulated_results[0]['output_layer_labels'].tolist()
      scores_file = os.path.join(
        self.results_dir, 'result_{}_scores.json'.format(
          global_step
        )
      )
      tp = np.sum([r['tp'] for r in accumulated_results], axis=0)
      fp = np.sum([r['fp'] for r in accumulated_results], axis=0)
      fn = np.sum([r['fn'] for r in accumulated_results], axis=0)
      tn = np.sum([r['tn'] for r in accumulated_results], axis=0)
      f1 = f1_from_tp_fp_fn(tp.astype(float), fp, fn)
      scores_str = json.dumps({
        'global_step': global_step,
        'accuracy': float(np.mean([r['accuracy'] for r in accumulated_results])),
        'output_layer_labels': output_layer_labels,
        'confusion_matrix': sum([r['confusion_matrix'] for r in accumulated_results]).tolist(),
        'tp': tp.tolist(),
        'fp': fp.tolist(),
        'fn': fn.tolist(),
        'tn': tn.tolist(),
        'f1': f1.tolist(),
        'micro_f1': float(np.mean([r['micro_f1'] for r in accumulated_results])),
        'macro_f1': float(np.mean(f1)),
        'count': sum([r['count'] for r in accumulated_results])
      }, indent=2)
      with FileIO(scores_file, 'w') as f:
        f.write(scores_str)

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
    num_detailed_eval_batches = min(10, num_eval_batches)
    if self.stream:
      for _ in range(num_eval_batches):
        session.run(tensors.metric_updates)
    else:
      if not self.batch_of_examples:
        for _ in range(num_eval_batches):
          self.batch_of_examples.append(session.run(tensors.examples))

      metric_values = None
      accumulated_results = None

      for eval_index in range(num_eval_batches):
        session.run(tensors.metric_updates, {
          tensors.examples: self.batch_of_examples[eval_index]
        })

        detailed_evaluation = eval_index < num_detailed_eval_batches

        fetches = self._get_default_fetches(tensors)
        self._add_evaluation_result_fetches(fetches, tensors)
        if detailed_evaluation:
          self._add_image_fetches(fetches, tensors)
        fetches['summary_value'] = tensors.summary
        self._check_fetches(fetches)
        results = session.run(fetches)

        accumulated_results = self._accumulate_evaluation_results(results, accumulated_results)
        if detailed_evaluation:
          self._save_prediction_summary_image(eval_index, results)
          self._save_result_images(eval_index, results)
          self._save_meta(eval_index, results)

          global_step = results['global_step']
          summary_value = results['summary_value']
          summary_writer.add_summary(summary_value, global_step)
          summary_writer.flush()

        metric_values = results['metric_values']

      self._save_accumulate_evaluation_results(accumulated_results)
      
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
    num_eval_batches = self.num_eval_batches
    num_detailed_eval_batches = min(10, num_eval_batches)
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

    accumulated_results = None

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

          detailed_evaluation = eval_index < num_detailed_eval_batches

          fetches = self._get_default_fetches(tensors)
          self._add_evaluation_result_fetches(fetches, tensors)
          if detailed_evaluation:
            self._add_image_fetches(fetches, tensors)
          self._check_fetches(fetches)
          results = session.run(fetches)

          accumulated_results = self._accumulate_evaluation_results(results, accumulated_results)
          if detailed_evaluation:
            self._save_prediction_summary_image(eval_index, results)
            self._save_result_images(eval_index, results)
            self._save_meta(eval_index, results)

          input_uri = results['input_uri']
          metric_values = results['metric_values']
          csv_f.write('{},{}\n'.format(input_uri, metric_values[0]))

        self._save_accumulate_evaluation_results(accumulated_results)
