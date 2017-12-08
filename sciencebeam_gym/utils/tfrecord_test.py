from six import text_type

from sciencebeam_gym.utils.tfrecord import (
  iter_examples_to_dict_list,
  dict_to_example
)

def dict_to_example_and_reverse(props):
  return list(iter_examples_to_dict_list([
    dict_to_example(props).SerializeToString()
  ]))[0]

def assert_dict_to_example_and_reverse(props):
  assert dict_to_example_and_reverse(props) == props

class TestDictToExampleAndIterExamplesToDictList(object):
  def test_should_handle_bytes(self):
    assert_dict_to_example_and_reverse({
      b'a': b'data'
    })

  def test_should_handle_unicode(self):
    assert_dict_to_example_and_reverse({
      b'a': text_type('data')
    })

  def test_should_handle_int(self):
    assert_dict_to_example_and_reverse({
      b'a': 1
    })
