from sciencebeam_gym.utils.tfrecord import (
    iter_examples_to_dict_list,
    dict_to_example
)


def dict_to_example_and_reverse(props):
    return list(iter_examples_to_dict_list([
        dict_to_example(props).SerializeToString()
    ]))[0]


class TestDictToExampleAndIterExamplesToDictList(object):
    def test_should_handle_bytes(self):
        props = {
            'a': b'data'
        }
        assert dict_to_example_and_reverse(props) == props

    def test_should_encode_text_as_bytes(self):
        props = {
            'a': u'data'
        }
        assert dict_to_example_and_reverse(props) == {
            'a': b'data'
        }

    def test_should_handle_int(self):
        props = {
            'a': 123
        }
        assert dict_to_example_and_reverse(props) == props
