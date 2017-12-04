import logging

import apache_beam as beam
from apache_beam.metrics.metric import Metrics

def get_logger():
  return logging.getLogger(__name__)

def Spy(f):
  def spy_wrapper(x):
    f(x)
    return x
  return spy_wrapper

def MapSpy(f):
  return beam.Map(Spy(f))

def MapOrLog(fn, log_fn=None, error_count=None):
  if log_fn is None:
    log_fn = lambda e, x: (
      get_logger().warning(
        'caught exception (ignoring item): %s, input: %.100s...',
        e, x, exc_info=e
      )
    )
  error_counter = (
    Metrics.counter('MapOrLog', error_count)
    if error_count
    else None
  )
  def wrapper(x):
    try:
      yield fn(x)
    except Exception as e:
      if error_counter:
        error_counter.inc()
      log_fn(e, x)
  return beam.FlatMap(wrapper)

LEVEL_MAP = {
  'info': logging.INFO,
  'debug': logging.DEBUG
}

def Count(name, counter_value_fn):
  counter = Metrics.counter('Count', name)
  def wrapper(x):
    counter.inc(counter_value_fn(x) if counter_value_fn else 1)
    return x
  return name >> beam.Map(wrapper)

class TransformAndCount(beam.PTransform):
  def __init__(self, transform, counter_name, counter_value_fn=None):
    super(TransformAndCount, self).__init__()
    self.transform = transform
    self.counter_name = counter_name
    self.counter_value_fn = counter_value_fn

  def expand(self, pcoll):
    return (
      pcoll |
      self.transform |
      "Count" >> Count(self.counter_name, self.counter_value_fn)
    )

class TransformAndLog(beam.PTransform):
  def __init__(self, transform, log_fn=None, log_prefix='', log_value_fn=None, log_level='info'):
    super(TransformAndLog, self).__init__()
    self.transform = transform
    if log_fn is None:
      if log_value_fn is None:
        log_value_fn = lambda x: x
      log_level = LEVEL_MAP.get(log_level, log_level)
      self.log_fn = lambda x: get_logger().log(
        log_level, '%s%.50s...', log_prefix, log_value_fn(x)
      )
    else:
      self.log_fn = log_fn

  def expand(self, pcoll):
    return (
      pcoll |
      self.transform |
      "Log" >> MapSpy(self.log_fn)
    )
