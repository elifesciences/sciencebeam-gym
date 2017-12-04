import pytest

import apache_beam as beam
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.testing.util import (
  assert_that,
  equal_to
)

from sciencebeam_gym.beam_utils.testing import (
  BeamTest,
  TestPipeline
)

from sciencebeam_gym.beam_utils.utils import (
  MapOrLog
)

SOME_VALUE_1 = 'value 1'
SOME_VALUE_CAUSING_EXCEPTION = 1

SOME_FN = lambda x: x.upper()
def FN_RAISING_EXCEPTION(_):
  raise RuntimeError('oh dear')

ERROR_COUNT_METRIC_NAME = 'error_count'

class TestMapOrLog(BeamTest):
  def test_should_pass_through_return_value_if_no_exception_was_raised(self):
    fn = lambda x: x.upper()
    with TestPipeline() as p:
      result = (
        p |
        beam.Create([SOME_VALUE_1]) |
        MapOrLog(SOME_FN)
      )
      assert_that(result, equal_to([fn(SOME_VALUE_1)]))

  def test_should_skip_entries_that_cause_an_exception(self):
    with TestPipeline() as p:
      result = (
        p |
        beam.Create([SOME_VALUE_1]) |
        MapOrLog(FN_RAISING_EXCEPTION)
      )
      assert_that(result, equal_to([]))

  def test_should_not_increase_error_metric_counter_if_no_exception_raised(self):
    with TestPipeline() as p:
      _ = (
        p |
        beam.Create([SOME_VALUE_1]) |
        MapOrLog(FN_RAISING_EXCEPTION, error_count=ERROR_COUNT_METRIC_NAME)
      )
      p_result = p.run()
      p_result.wait_until_finish()
      error_counter = p_result.metrics().query(
        MetricsFilter().with_name(ERROR_COUNT_METRIC_NAME)
      )['counters']
      assert len(error_counter) == 1
      assert error_counter[0].committed == 1

  def test_should_increase_error_metric_counter_if_exception_was_raised(self):
    with TestPipeline() as p:
      _ = (
        p |
        beam.Create([SOME_VALUE_1]) |
        MapOrLog(FN_RAISING_EXCEPTION, error_count=ERROR_COUNT_METRIC_NAME)
      )
      p_result = p.run()
      p_result.wait_until_finish()
      error_counter = p_result.metrics().query(
        MetricsFilter().with_name(ERROR_COUNT_METRIC_NAME)
      )['counters']
      assert len(error_counter) == 1
      assert error_counter[0].committed == 1
