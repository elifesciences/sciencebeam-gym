from six import StringIO

from sciencebeam_gym.preprocess.color_map import (
    parse_color_map_from_file
)


class TestParseColorMapFromFile(object):
    def test_should_parse_rgb_color_values(self):
        data = '\n'.join([
            '[color_map]',
            'tag1 = (255, 0, 0)'
        ])
        color_map = parse_color_map_from_file(StringIO(data))
        assert color_map == {
            'tag1': (255, 0, 0)
        }
