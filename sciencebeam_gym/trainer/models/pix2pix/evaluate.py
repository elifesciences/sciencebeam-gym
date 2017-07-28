from __future__ import absolute_import
from __future__ import division

import collections

import tensorflow as tf

EvaluationTensors = collections.namedtuple(
  "EvaluationTensors", [
    "confusion_matrix",
    "tp",
    "fp",
    "fn",
    "accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1"
  ]
)

def output_probabilities_to_class(outputs):
  return tf.argmax(outputs, 3)

def to_1d_vector(tensor):
  return tf.reshape(tensor, [-1])

def _evaluate_from_confusion_matrix(confusion, accuracy=None):
  actual_p = tf.reduce_sum(confusion, axis=0)
  pred_p = tf.reduce_sum(confusion, axis=1)
  tp = tf.diag_part(confusion)
  fp = actual_p - tp
  fn = pred_p - tp
  total_tp = tf.reduce_sum(tp)
  total_fp = tf.reduce_sum(fp)
  total_fn = tf.reduce_sum(fn)
  # Note: micro averages (with equal weights) will lead to the same precision, recall, f1
  micro_precision = total_tp / (total_tp + total_fp)
  micro_recall = total_tp / (total_tp + total_fn)
  micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
  return EvaluationTensors(
    confusion_matrix=confusion,
    tp=tp,
    fp=fp,
    fn=fn,
    accuracy=accuracy,
    micro_precision=micro_precision,
    micro_recall=micro_recall,
    micro_f1=micro_f1
  )

def evaluate_predictions(labels, predictions, n_classes, has_unknown_class=False):
  correct_prediction = tf.equal(labels, predictions)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  confusion = tf.contrib.metrics.confusion_matrix(
    labels=labels,
    predictions=predictions,
    num_classes=n_classes
  )

  if has_unknown_class:
    # remove unknown class
    confusion = tf.slice(confusion, [0, 0], [int(n_classes - 1), int(n_classes - 1)])

  return _evaluate_from_confusion_matrix(
    confusion=confusion,
    accuracy=accuracy
  )

def evaluate_separate_channels(targets, outputs, has_unknown_class=False):
  n_classes = targets.shape[-1]

  labels = to_1d_vector(output_probabilities_to_class(targets))
  predictions = to_1d_vector(output_probabilities_to_class(outputs))
  return evaluate_predictions(
    labels=labels,
    predictions=predictions,
    n_classes=n_classes,
    has_unknown_class=has_unknown_class
  )


def evaluation_summary(evaluation_tensors):
  tf.summary.scalar("micro_precision", evaluation_tensors.micro_precision)
  tf.summary.scalar("micro_recall", evaluation_tensors.micro_recall)
  tf.summary.scalar("micro_f1", evaluation_tensors.micro_f1)
  tf.summary.scalar("accuracy", evaluation_tensors.accuracy)
