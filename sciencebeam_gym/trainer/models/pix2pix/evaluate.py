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
    "tn",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1"
  ]
)

def output_probabilities_to_class(outputs):
  return tf.argmax(outputs, 3)

def to_1d_vector(tensor):
  return tf.reshape(tensor, [-1])

def precision_from_tp_fp(tp, fp):
  return tp / (tp + fp)

def recall_from_tp_fn(tp, fn):
  return tp / (tp + fn)

def f1_from_precision_recall(precision, recall):
  return 2 * precision * recall / (precision + recall)

def _evaluate_from_confusion_matrix(confusion, accuracy=None):
  total = tf.reduce_sum(confusion)
  actual_p = tf.reduce_sum(confusion, axis=0)
  pred_p = tf.reduce_sum(confusion, axis=1)
  tp = tf.diag_part(confusion)
  fp = actual_p - tp
  fn = pred_p - tp
  tn = total - tp - fp - fn
  precision = precision_from_tp_fp(tp, fp)
  recall = recall_from_tp_fn(tp, fn)
  f1 = f1_from_precision_recall(precision, recall)
  total_tp = tf.reduce_sum(tp)
  total_fp = tf.reduce_sum(fp)
  total_fn = tf.reduce_sum(fn)
  # Note: micro averages (with equal weights) will lead to the same precision, recall, f1
  micro_precision = precision_from_tp_fp(total_tp, total_fp)
  micro_recall = recall_from_tp_fn(total_tp, total_fn)
  micro_f1 = f1_from_precision_recall(micro_precision, micro_recall)
  macro_precision = tf.reduce_sum(precision)
  macro_recall = tf.reduce_sum(recall)
  macro_f1 = tf.reduce_sum(f1)
  return EvaluationTensors(
    confusion_matrix=confusion,
    tp=tp,
    fp=fp,
    fn=fn,
    tn=tn,
    precision=precision,
    recall=recall,
    f1=f1,
    accuracy=accuracy,
    micro_precision=micro_precision,
    micro_recall=micro_recall,
    micro_f1=micro_f1,
    macro_precision=macro_precision,
    macro_recall=macro_recall,
    macro_f1=macro_f1
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


def evaluation_summary(evaluation_tensors, layer_labels):
  tf.summary.scalar("micro_precision", evaluation_tensors.micro_precision)
  tf.summary.scalar("micro_recall", evaluation_tensors.micro_recall)
  tf.summary.scalar("micro_f1", evaluation_tensors.micro_f1)
  tf.summary.scalar("macro_f1", evaluation_tensors.macro_f1)
  tf.summary.scalar("accuracy", evaluation_tensors.accuracy)
  for i, layer_label in enumerate(layer_labels):
    tf.summary.scalar("f1_{}_{}".format(i, layer_label), evaluation_tensors.f1[i])
