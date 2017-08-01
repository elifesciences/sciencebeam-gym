# partially copied from tensorflow example project
from __future__ import absolute_import

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from multiprocessing import Pool

import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io

from sciencebeam_gym.trainer.evaluator import Evaluator
from sciencebeam_gym.trainer.util import (
  CustomSupervisor,
  SimpleStepScheduler,
  override_if_not_in_args
)

def get_logger():
  return logging.getLogger(__name__)

class TrainingProgressLogger(object):
  def __init__(self, start_time, start_step, task):
    self.start_time = start_time
    self.start_step = start_step
    self.last_log_time = start_time
    self.last_global_step = start_step
    self.last_local_step = 0
    self.task = task

  def get_last_log_time(self):
    return self.last_log_time

  def log(self, now, global_step, local_step):
    """Logs training progress."""
    logging.info(
      'Train [%s/%d], step %d (%.3f sec) %.1f '
      'global steps/s, %.1f local steps/s',
      self.task.type,
      self.task.index,
      global_step,
      (now - self.start_time),
      (global_step - self.last_global_step) /
      (now - self.last_log_time),
      (local_step - self.last_local_step) /
      (now - self.last_log_time)
    )
    self.last_log_time = now
    self.last_global_step = global_step
    self.last_local_step = local_step

def get_quantitative_evaluator(args, model, run_async):
  if args.quantitative_data_paths:
    return Evaluator(
      args,
      model,
      train_dir(args.output_path),
      args.quantitative_data_paths,
      dataset='quantitative_set',
      eval_set_size=args.quantitative_set_size or args.eval_set_size,
      quantitative_set_size=args.quantitative_set_size,
      run_async=run_async
    )
  else:
    return None

class Trainer(object):
  """Performs model training and optionally evaluation."""

  def __init__(self, args, model, cluster, task):
    self.args = args
    self.model = model
    self.cluster = cluster
    self.task = task
    self.run_async = None
    default_run_async = lambda f, args: f(*args)
    run_async = lambda f, args: (self.run_async or default_run_async)(f, args)
    self.evaluator = Evaluator(
      self.args, self.model,
      train_dir(self.args.output_path),
      self.args.eval_data_paths,
      'eval_set',
      run_async=run_async
    )
    self.train_evaluator = Evaluator(
      self.args, self.model,
      train_dir(self.args.output_path),
      self.args.train_data_paths,
      'train_set',
      run_async=run_async
    )
    self.quantitative_evaluator = get_quantitative_evaluator(
      self.args,
      self.model,
      run_async=run_async
    )
    self.min_train_eval_rate = args.min_train_eval_rate
    self.global_step = None
    self.last_save = 0

  def run_training(self):
    pool = Pool(processes=self.args.pool_size)
    self.run_async = lambda f, args: pool.apply_async(f, args)
    self._do_run_training()
    get_logger().info('Waiting for tasks to complete')
    pool.close()
    pool.join()
    self.run_async = None

  def _do_run_training(self):
    """Runs a Master."""
    logger = get_logger()
    self.train_evaluator.init()
    self.evaluator.init()
    if self.quantitative_evaluator:
      self.quantitative_evaluator.init()
    ensure_output_path(self.args.output_path)
    train_path = train_dir(self.args.output_path)
    # model_path = model_dir(self.args.output_path)
    is_master = self.task.type != 'worker'
    log_interval = self.args.log_interval_secs
    save_interval = self.args.save_interval_secs
    eval_interval = self.args.eval_interval_secs
    summary_interval = log_interval
    summary_freq = self.args.log_freq
    if is_master and self.task.index > 0:
      raise StandardError('Only one replica of master expected')

    if self.cluster:
      logging.info('Starting %s/%d', self.task.type, self.task.index)
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
        ps_device='/job:ps',
        worker_device='/job:%s/task:%d' % (self.task.type, self.task.index),
        cluster=self.cluster
      )
      # We use a device_filter to limit the communication between this job
      # and the parameter servers, i.e., there is no need to directly
      # communicate with the other workers; attempting to do so can result
      # in reliability problems.
      device_filters = [
        '/job:ps', '/job:%s/task:%d' % (self.task.type, self.task.index)
      ]
      config = tf.ConfigProto(device_filters=device_filters)
    else:
      target = ''
      device_fn = ''
      config = None

    logger.info('batch_size: %s', self.args.batch_size)

    with tf.Graph().as_default() as graph:
      with tf.device(device_fn):
        # Build the training graph.
        tensors = self.model.build_train_graph(
          self.args.train_data_paths,
          self.args.batch_size)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(
          max_to_keep=self.args.save_max_to_keep
        )

    # Create a "supervisor", which oversees the training process.
    sv = CustomSupervisor(
      model=self.model,
      graph=graph,
      is_chief=is_master,
      logdir=train_path,
      saver=saver,
      # Write summary_ops by hand.
      summary_op=None,
      global_step=tensors.global_step
    )

    save_path = sv.save_path

    should_retry = True
    local_step = 0

    while should_retry:
      try:
        should_retry = False
        with sv.managed_session(target, config=config) as session:
          start_time = time.time()
          now = start_time

          global_step = session.run(tensors.global_step)
          training_progress_logger = TrainingProgressLogger(
            start_time,
            global_step,
            self.task
          )

          log_scheduler = SimpleStepScheduler(
            lambda: training_progress_logger.log(now, global_step, local_step),
            min_interval=log_interval,
            min_freq=self.args.log_freq,
            step=global_step,
            last_run=start_time
          )

          def do_save():
            logger.info('saving model to %s (%s)', save_path, global_step)
            saver.save(session, save_path, tensors.global_step)

          save_scheduler = SimpleStepScheduler(
            do_save,
            min_interval=save_interval,
            min_freq=self.args.save_freq,
            step=global_step,
            last_run=start_time
          )

          eval_train_scheduler = SimpleStepScheduler(
            lambda: self.eval_train(session, tensors, global_step),
            min_interval=eval_interval,
            min_freq=self.args.eval_freq,
            step=global_step,
            last_run=start_time
          )

          schedulers = [
            log_scheduler,
            save_scheduler,
            eval_train_scheduler
          ]

          if is_master:
            eval_scheduler = SimpleStepScheduler(
              lambda: self.eval(global_step=global_step),
              min_interval=save_interval,
              min_freq=self.args.save_freq,
              step=global_step,
              last_run=start_time
            )
            schedulers = schedulers + [eval_scheduler]

          summary_op = sv.summary_op if tensors.summary is None else tensors.summary
          if summary_op is not None:
            schedulers.append(SimpleStepScheduler(
              lambda: sv.summary_writer.add_summary(
                *session.run([summary_op, tensors.global_step])
              ),
              min_interval=summary_interval,
              min_freq=summary_freq,
              step=global_step,
              last_run=start_time
            ))

          # Loop until the supervisor shuts down or args.max_steps have
          # completed.
          max_steps = self.args.max_steps
          while not sv.should_stop() and global_step < max_steps:
            logging.info("global_step: %s", global_step)
            try:
              # Run one step of the model.
              global_step = session.run([tensors.global_step, tensors.train])[0]
              logging.info("global_step: %s", global_step)
              local_step += 1

              now = time.time()
              for scheduler in schedulers:
                scheduler.step(now)

            except tf.errors.AbortedError as e:
              should_retry = True
              logging.info('AbortedError (%s)', e)
            except (KeyboardInterrupt, tf.errors.CancelledError):
              logging.info('cancelled')
              should_retry = False

          logging.info('finished (is_master: %s)', is_master)

          if is_master:
            # Take the final checkpoint and compute the final accuracy.
            now = time.time()
            for scheduler in schedulers:
              scheduler.flush(now)

      except tf.errors.AbortedError as e:
        should_retry = True
        logging.info('AbortedError (%s)', e)

    # Ask for all the services to stop.
    sv.stop()

  def eval_train(self, session, tensors, global_step):
    """Runs evaluation loop."""
    logging.info(
      'Eval, step %d:\n- on train set %s',
      global_step,
      self.model.format_metric_values(
        self.train_evaluator.evaluate_in_session(
          session=session,
          tensors=tensors
        )
      )
    )

  def eval(self, global_step=None):
    """Runs evaluation loop."""
    if self.quantitative_evaluator:
      logging.info(
        'Quantitive Eval, step %s:\n- on eval set %s',
        global_step,
        self.model.format_metric_values(self.quantitative_evaluator.evaluate())
      )
    logging.info(
      'Eval, step %s:\n- on eval set %s',
      global_step,
      self.model.format_metric_values(self.evaluator.evaluate())
    )

def copy_data_to_tmp(input_files):
  """Copies data to /tmp/ and returns glob matching the files."""
  files = []
  for e in input_files:
    for path in e.split(','):
      files.extend(file_io.get_matching_files(path))

  for path in files:
    if not path.startswith('gs://'):
      return input_files

  tmp_path = os.path.join('/tmp/', str(uuid.uuid4()))
  os.makedirs(tmp_path)
  subprocess.check_call(['gsutil', '-m', '-q', 'cp', '-r'] + files + [tmp_path])
  return [os.path.join(tmp_path, '*')]


def write_predictions(args, model, cluster, task):
  if not cluster or not task or task.type == 'master':
    pass  # Run locally.
  else:
    raise ValueError('invalid task_type %s' % (task.type,))

  logger = get_logger()
  logger.info('Starting to write predictions on %s/%d', task.type, task.index)
  pool = Pool(processes=args.pool_size)
  run_async = lambda f, args: pool.apply_async(f, args)

  quantitative_evaluator = get_quantitative_evaluator(
    args,
    model,
    run_async=run_async
  )
  if quantitative_evaluator:
    quantitative_evaluator.init()
    quantitative_evaluator.write_predictions()

  evaluator = Evaluator(
    args, model, train_dir(args.output_path), args.eval_data_paths,
    run_async=run_async
  )
  evaluator.init()
  evaluator.write_predictions()

  logger.info('Waiting for background tasks to finish')
  pool.close()
  pool.join()
  logger.info('Done writing predictions on %s/%d', task.type, task.index)


def dispatch(args, model, cluster, task):
  if not cluster or not task or task.type == 'master':
    # Run locally.
    Trainer(args, model, cluster, task).run_training()
  elif task.type == 'ps':
    run_parameter_server(cluster, task)
  elif task.type == 'worker':
    Trainer(args, model, cluster, task).run_training()
  else:
    raise ValueError('invalid task_type %s' % (task.type,))


def run_parameter_server(cluster, task):
  logging.info('Starting parameter server %d', task.index)
  server = start_server(cluster, task)
  server.join()


def start_server(cluster, task):
  if not task.type:
    raise ValueError('--task_type must be specified.')
  if task.index is None:
    raise ValueError('--task_index must be specified.')

  # Create and start a server.
  return tf.train.Server(
    tf.train.ClusterSpec(cluster),
    protocol='grpc',
    job_name=task.type,
    task_index=task.index
  )


def ensure_output_path(output_path):
  if not output_path:
    raise ValueError('output_path must be specified')

  # GCS doesn't have real directories.
  if output_path.startswith('gs://'):
    return

  ensure_dir(output_path)


def ensure_dir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    # If the directory already existed, ignore the error.
    if e.args[0] == 17:
      pass
    else:
      raise


def train_dir(output_path):
  return os.path.join(output_path, 'train')


def eval_dir(output_path):
  return os.path.join(output_path, 'eval')


def model_dir(output_path):
  return os.path.join(output_path, 'model')

def run(model, argv):
  """Runs the training loop."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--train_data_paths',
    type=str,
    action='append',
    help='The paths to the training data files. '
    'Can be comma separated list of files or glob pattern.'
  )
  parser.add_argument(
    '--eval_data_paths',
    type=str,
    action='append',
    help='The path to the files used for evaluation. '
    'Can be comma separated list of files or glob pattern.'
  )
  parser.add_argument(
    '--quantitative_data_paths',
    type=str,
    action='append',
    help='The path to the files used for quantitative evaluation. '
    'You may choose a different set for the quantitative analysis to keep the results consistent.'
  )
  parser.add_argument(
    '--output_path',
    type=str,
    help='The path to which checkpoints and other outputs '
    'should be saved. This can be either a local or GCS '
    'path.'
  )
  parser.add_argument(
    '--max_steps',
    type=int,
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    help='Number of examples to be processed per mini-batch.'
  )
  parser.add_argument(
    '--eval_set_size', type=int, help='Number of examples in the eval set.'
  )
  parser.add_argument(
    '--quantitative_set_size',
    type=int,
    help='Number of examples in the quantitative eval set.'
  )
  parser.add_argument(
    '--eval_batch_size', type=int, help='Number of examples per eval batch.'
  )
  parser.add_argument(
    '--eval_interval_secs',
    type=float,
    default=5,
    help='Minimal interval between calculating evaluation metrics and saving'
    ' evaluation summaries.'
  )
  parser.add_argument(
    '--eval_freq',
    type=int,
    default=100,
    help='Frequancy in steps between calculating evaluation metrics and saving'
    ' evaluation summaries.'
  )
  parser.add_argument(
    '--save_interval_secs',
    type=float,
    default=300,
    help='Minimal interval between saving the model checkpoint'
  )
  parser.add_argument(
    '--save_freq',
    type=int,
    default=1000,
    help='Frequancy in steps between saving the model checkpoint'
  )
  parser.add_argument(
    '--save_max_to_keep',
    type=int,
    default=2,
    help='Maximum number of recent checkpoint files to keep'
  )
  parser.add_argument(
    '--log_interval_secs',
    type=float,
    default=5,
    help='Minimal interval between logging training metrics and saving '
    'training summaries.'
  )
  parser.add_argument(
    '--log_freq',
    type=int,
    default=500,
    help='Frequancy in steps between logging training metrics and saving '
    'training summaries.'
  )
  parser.add_argument(
    '--write_predictions',
    action='store_true',
    default=False,
    help='If set, model is restored from latest checkpoint '
    'and predictions are written to a csv file and no training is performed.'
  )
  parser.add_argument(
    '--min_train_eval_rate',
    type=int,
    default=20,
    help='Minimal train / eval time ratio on master. '
    'Default value 20 means that 20x more time is used for training than '
    'for evaluation. If evaluation takes more time the eval_interval_secs '
    'is increased.'
  )
  parser.add_argument(
    '--write_to_tmp',
    action='store_true',
    default=False,
    help='If set, all checkpoints and summaries are written to '
    'local filesystem (/tmp/) and copied to gcs once training is done. '
    'This can speed up training but if training job fails all the summaries '
    'and checkpoints are lost.'
  )
  parser.add_argument(
    '--copy_train_data_to_tmp',
    action='store_true',
    default=False,
    help='If set, training data is copied to local filesystem '
    '(/tmp/). This can speed up training but requires extra space on the '
    'local filesystem.'
  )
  parser.add_argument(
    '--copy_eval_data_to_tmp',
    action='store_true',
    default=False,
    help='If set, evaluation data is copied to local filesystem '
    '(/tmp/). This can speed up training but requires extra space on the '
    'local filesystem.'
  )
  parser.add_argument(
    '--streaming_eval',
    action='store_true',
    default=False,
    help='If set to True the evaluation is performed in streaming mode. '
    'During each eval cycle the evaluation data is read and parsed from '
    'files. This allows for having very large evaluation set. '
    'If set to False (default) evaluation data is read once and cached in '
    'memory. This results in faster evaluation cycle but can potentially '
    'use more memory (in streaming mode large per-file read-ahead buffer is '
    'used - which may exceed eval data size).'
  )
  parser.add_argument(
    '--pool_size',
    type=int,
    default=50,
    help='Number of examples in the eval set.'
  )

  args, _ = parser.parse_known_args(argv)

  env = json.loads(os.environ.get('TF_CONFIG', '{}'))

  # Print the job data as provided by the service.
  logging.info('Original job data: %s', env.get('job', {}))

  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task = type('TaskSpec', (object,), task_data)
  trial = task_data.get('trial')
  if trial is not None:
    args.output_path = os.path.join(args.output_path, trial)
  if args.write_to_tmp and args.output_path.startswith('gs://'):
    output_path = args.output_path
    args.output_path = os.path.join('/tmp/', str(uuid.uuid4()))
    os.makedirs(args.output_path)
  else:
    output_path = None

  if args.copy_train_data_to_tmp:
    args.train_data_paths = copy_data_to_tmp(args.train_data_paths)
  if args.copy_eval_data_to_tmp:
    args.eval_data_paths = copy_data_to_tmp(args.eval_data_paths)

  if not args.eval_batch_size:
    # If eval_batch_size not set, use min of batch_size and eval_set_size
    args.eval_batch_size = min(args.batch_size, args.eval_set_size)
    logging.info("setting eval batch size to %s", args.eval_batch_size)

  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  if args.write_predictions:
    write_predictions(args, model, cluster, task)
  else:
    dispatch(args, model, cluster, task)

  if output_path and (not cluster or not task or task.type == 'master'):
    subprocess.check_call([
        'gsutil', '-m', '-q', 'cp', '-r', args.output_path + '/*', output_path
    ])
    shutil.rmtree(args.output_path, ignore_errors=True)

def get_model_factory(model_name):
  if model_name == 'pix2pix':
    import sciencebeam_gym.trainer.models.pix2pix.pix2pix_model as model_factory
    return model_factory
  raise Exception('unsupported model: {}'.format(model_name))

def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='The name of the model'
  )
  args, other_args = parser.parse_known_args()

  model_factory = get_model_factory(args.model)

  model, task_args = model_factory.create_model(other_args)
  override_if_not_in_args('--max_steps', '1000', task_args)
  override_if_not_in_args('--batch_size', '100', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)
  run(model, task_args)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
