import os
import logging
import json
from io import BytesIO

import numpy as np

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from PIL import Image

from sciencebeam_gym.utils.pyplot import pyplot as plt

from sciencebeam_gym.utils.tf import (
    FileIO
)

from sciencebeam_gym.trainer.util import (
    CustomSupervisor,
    get_graph_size
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
        ax.imshow(np.dstack((image.astype(np.uint8),) * 3) * 100, aspect='auto')
    ax.set_title(label, color=(0.5, 0.5, 0.5), y=0.995)
    ax.set_axis_off()
    ax.set(xlim=[0, 255], ylim=[255, 0], aspect=1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


def show_result_images3(input_image, annot, output_image):
    figsize = plt.figaspect(1.1 / 3.0)
    fig, (ax_img, ax_annot, ax_out) = plt.subplots(
        1,
        3,
        sharey=True,
        figsize=figsize,
        frameon=False,
        facecolor=None,
        dpi=60
    )

    plot_image(ax_img, input_image, 'input')
    plot_image(ax_annot, annot, 'target')
    plot_image(ax_out, output_image, 'prediction')

    margin = 0.01
    fig.subplots_adjust(
        left=margin,
        right=1.0 - margin,
        top=0.95 - margin,
        bottom=margin,
        wspace=0.05
    )

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


def to_list_if_not_none(x):
    return x.tolist() if x is not None else x


IMAGE_PREFIX = 'image_'


class Evaluator(object):
    """Loads variables from latest checkpoint and performs model evaluation."""

    def __init__(
            self, args, model,
            checkpoint_path,
            data_paths,
            dataset='eval',
            eval_batch_size=None,
            eval_set_size=None,
            qualitative_set_size=None,
            run_async=None):

        self.eval_batch_size = eval_batch_size or args.eval_batch_size
        self.num_eval_batches = (eval_set_size or args.eval_set_size) // self.eval_batch_size
        self.num_detail_eval_batches = (
            min((qualitative_set_size or 10), args.eval_set_size) // self.eval_batch_size
        )
        self.checkpoint_path = checkpoint_path
        self.output_path = os.path.join(args.output_path, dataset)
        self.eval_data_paths = data_paths
        self.batch_size = args.batch_size
        self.stream = args.streaming_eval
        self.model = model
        self.results_dir = os.path.join(self.output_path, 'results')
        self.graph_size = None
        self.run_async = run_async
        if not run_async:
            self.run_async = lambda f, args: f(*args)

    def init(self):
        file_io.recursive_create_dir(self.results_dir)

    def _check_fetches(self, fetches: dict):
        for k, v in fetches.items():
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

    def _add_image_fetches(self, fetches: dict, tensors):
        for k, v in tensors.image_tensors.items():
            fetches[IMAGE_PREFIX + k] = v
        return fetches

    def _add_evaluation_result_fetches(self, fetches, tensors):
        if tensors.evaluation_result:
            if tensors.output_layer_labels is not None:
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
        if results.get('confusion_matrix') is None:
            return accumulated_results
        if accumulated_results is None:
            accumulated_results = []
        accumulated_results.append({
            'output_layer_labels': results.get('output_layer_labels'),
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
            first_result = accumulated_results[0]
            global_step = first_result['global_step']
            output_layer_labels = to_list_if_not_none(first_result.get('output_layer_labels'))
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
            meta = {
                'global_step': global_step,
                'batch_size': self.batch_size
            }
            if self.graph_size:
                meta['graph_size'] = self.graph_size
            scores = {
                'accuracy': float(np.mean([r['accuracy'] for r in accumulated_results])),
                'output_layer_labels': output_layer_labels,
                'confusion_matrix': sum(
                    [r['confusion_matrix'] for r in accumulated_results]
                ).tolist(),
                'tp': to_list_if_not_none(tp),
                'fp': to_list_if_not_none(fp),
                'fn': to_list_if_not_none(fn),
                'tn': to_list_if_not_none(tn),
                'f1': to_list_if_not_none(f1),
                'micro_f1': float(np.mean([r['micro_f1'] for r in accumulated_results])),
                'macro_f1': float(np.mean(f1)),
                'count': sum([r['count'] for r in accumulated_results]),
                'meta': meta
            }
            scores_str = json.dumps(scores, indent=2)
            with FileIO(scores_file, 'w') as f:
                f.write(scores_str)

    def _save_prediction_summary_image_for(
            self, eval_index, global_step, inputs, targets, outputs, name):

        for batch_index, input_image, target_image, output_image in zip(
            range(len(inputs)), inputs, targets, outputs
        ):

            fig = show_result_images3(
                input_image,
                target_image,
                output_image
            )
            result_file = os.path.join(
                self.results_dir, 'result_{}_{}_{}_{}.png'.format(
                    global_step, eval_index, batch_index, name
                )
            )
            logging.info('result_file: %s', result_file)
            bio = BytesIO()
            plt.savefig(bio, format='png', transparent=False, frameon=True, dpi='figure')
            plt.close(fig)
            self.run_async(save_file, (result_file, bio.getvalue()))

    def _save_prediction_summary_image(self, eval_index, results):
        global_step = results['global_step']
        self._save_prediction_summary_image_for(
            eval_index,
            global_step,
            results['input_image'],
            results['annotation_image'],
            results['output_image'],
            'summary_output'
        )

        if results.get(IMAGE_PREFIX + 'outputs_most_likely') is not None:
            self._save_prediction_summary_image_for(
                eval_index,
                global_step,
                results['input_image'],
                results['annotation_image'],
                results[IMAGE_PREFIX + 'outputs_most_likely'],
                'summary_most_likely'
            )
        outputs_key_needle = '_outputs_'
        for k in results.keys():
            outputs_key_needle_index = k.find(outputs_key_needle)
            if k.startswith(IMAGE_PREFIX) and outputs_key_needle_index >= 0:
                targets_key = k.replace(outputs_key_needle, '_targets_')
                if targets_key not in results:
                    continue
                self._save_prediction_summary_image_for(
                    eval_index,
                    global_step,
                    results['input_image'],
                    results[targets_key],
                    results[k],
                    'summary_' + k[(outputs_key_needle_index + len(outputs_key_needle)):]
                )

    def _save_result_images(self, eval_index, results):
        global_step = results['global_step']
        for k in results.keys():
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
                'global_step': int(global_step),
                'eval_index': eval_index,
                'batch_index': batch_index,
                'metric_values': [float(x) for x in metric_values],
                'input_uri': str(input_uri)
            }, indent=2)
            with FileIO(meta_file, 'w') as meta_f:
                meta_f.write(meta_str)

    def evaluate_in_session(self, session, tensors, num_eval_batches=None):
        summary_writer = tf.summary.FileWriter(self.output_path)
        num_eval_batches = num_eval_batches or self.num_eval_batches
        num_detailed_eval_batches = min(self.num_detail_eval_batches, num_eval_batches)
        if self.stream:
            for _ in range(num_eval_batches):
                session.run(tensors.metric_updates, feed_dict={
                    tensors.is_training: False
                })
        else:
            get_logger().info('tensors.examples: %s', tensors.examples)

            metric_values = None
            accumulated_results = None

            for eval_index in range(num_eval_batches):
                detailed_evaluation = eval_index < num_detailed_eval_batches

                fetches = self._get_default_fetches(tensors)
                self._add_evaluation_result_fetches(fetches, tensors)
                if detailed_evaluation:
                    self._add_image_fetches(fetches, tensors)
                fetches['summary_value'] = tensors.summary
                self._check_fetches(fetches)
                results = session.run(fetches, feed_dict={
                    tensors.is_training: False
                })

                accumulated_results = self._accumulate_evaluation_results(
                    results, accumulated_results)
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

        metric_values = session.run(tensors.metric_values, feed_dict={
            tensors.is_training: False
        })
        return metric_values

    def evaluate(self, num_eval_batches=None):
        """Run one round of evaluation, return loss and accuracy."""

        num_eval_batches = num_eval_batches or self.num_eval_batches
        with tf.Graph().as_default() as graph:
            tensors = self.model.build_eval_graph(
                self.eval_data_paths,
                self.eval_batch_size
            )
            self.graph_size = get_graph_size()

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
                        session.run(tensors.metric_updates, feed_dict={
                            tensors.is_training: False
                        })
                else:
                    logging.info('start queue runners (batch)')
                    sv.start_queue_runners(session)

                logging.info('evaluate_in_session')
                return self.evaluate_in_session(session, tensors)
        finally:
            sv.stop()

    def write_predictions(self):
        """Run one round of predictions and write predictions to csv file."""
        num_eval_batches = self.num_eval_batches
        num_detailed_eval_batches = self.num_detail_eval_batches
        with tf.Graph().as_default() as graph:
            tensors = self.model.build_eval_graph(
                self.eval_data_paths,
                self.batch_size
            )
            self.graph_size = get_graph_size()
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
                    results = session.run(fetches, feed_dict={
                        tensors.is_training: False
                    })

                    accumulated_results = self._accumulate_evaluation_results(
                        results, accumulated_results)
                    if detailed_evaluation:
                        self._save_prediction_summary_image(eval_index, results)
                        self._save_result_images(eval_index, results)
                        self._save_meta(eval_index, results)

                    input_uri = results['input_uri']
                    metric_values = results['metric_values']
                    csv_f.write('{},{}\n'.format(input_uri, metric_values[0]))

                self._save_accumulate_evaluation_results(accumulated_results)
