from sciencebeam_utils.utils.xml import (
    set_or_remove_attrib
)

from sciencebeam_gym.utils.bounding_box import (
    BoundingBox
)

from sciencebeam_gym.structured_document import (
    AbstractStructuredDocument,
    get_scoped_attrib_name,
    get_attrib_by_scope
)

TAG_ATTRIB_NAME = 'tag'


def get_node_bounding_box(t):
    return BoundingBox(
        float(t.attrib.get('x', 0)),
        float(t.attrib.get('y', 0)),
        float(t.attrib['width']),
        float(t.attrib['height'])
    )


def _get_tag_attrib_name(scope, level):
    return get_scoped_attrib_name(TAG_ATTRIB_NAME, scope=scope, level=level)


class LxmlStructuredDocument(AbstractStructuredDocument):
    def __init__(self, root):
        self.root = root

    def get_pages(self):
        return self.root.findall('.//PAGE')

    def get_lines_of_page(self, page):
        return page.findall('.//TEXT')

    def get_tokens_of_line(self, line):
        return line.findall('./TOKEN')

    def get_x(self, parent):
        return parent.attrib.get('x')

    def get_text(self, parent):
        return parent.text

    def get_tag(self, parent, scope=None, level=None):
        return parent.attrib.get(_get_tag_attrib_name(scope, level))

    def set_tag(self, parent, tag, scope=None, level=None):
        set_or_remove_attrib(parent.attrib, _get_tag_attrib_name(scope, level), tag)

    def get_tag_by_scope(self, parent):
        return get_attrib_by_scope(parent.attrib, TAG_ATTRIB_NAME)

    def get_bounding_box(self, parent):
        return get_node_bounding_box(parent)

    def set_bounding_box(self, parent, bounding_box):
        parent.attrib['x'] = str(bounding_box.x)
        parent.attrib['y'] = str(bounding_box.y)
        parent.attrib['width'] = str(bounding_box.width)
        parent.attrib['height'] = str(bounding_box.height)
