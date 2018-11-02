import hashlib

from lxml import etree

from sciencebeam_utils.utils.collection import flatten

from sciencebeam_gym.structured_document.svg import (
    SVG_TAG_ATTRIB
)

DEFAULT_COLORS = [
    'maroon', 'red', 'purple', '#c0c', 'green', '#0c0',
    'olive', '#cc0', 'navy', 'blue', 'teal', '#0cc'
]
# colors replaced due readability issues:
# fuchsia, lime, yellow, aqua


def color_for_tag(tag, colors=None):
    if colors is None:
        colors = DEFAULT_COLORS
    h = int(hashlib.md5(tag.encode('utf8')).hexdigest(), 16)
    return colors[h % len(colors)]


def style_props_for_tag(tag):
    return {
        'fill': color_for_tag(tag)
    }


def style_props_for_tags(tags):
    return {
        tag: style_props_for_tag(tag)
        for tag in tags
    }


def render_style_props(style_props):
    return '\n'.join([
        '  {}: {}'.format(k, v)
        for k, v in style_props.items()
    ])


def style_block_for_tag(tag, style_props):
    return 'text[{tag_attrib}~="{tag}"] {{\n{content}\n}}'.format(
        tag_attrib=SVG_TAG_ATTRIB,
        tag=tag,
        content=render_style_props(style_props)
    )


def style_block_for_tags(tags):
    style_props_map = style_props_for_tags(tags)
    return '\n\n'.join([
        style_block_for_tag(tag, style_props_map[tag])
        for tag in tags
    ])


def tags_for_node(node):
    svga_tags = node.attrib.get(SVG_TAG_ATTRIB, '').strip()
    if len(svga_tags) == 0:
        return []
    return svga_tags.split(' ')


def tags_for_nodes(nodes):
    return sorted(set(flatten([
        tags_for_node(node)
        for node in nodes
    ])))


def nodes_with_tags(svg_root):
    for node in svg_root.findall('*[@{}]'.format(SVG_TAG_ATTRIB)):
        yield node
        for nested_node in nodes_with_tags(node):
            yield nested_node


def add_title_to_nodes(nodes):
    for node in nodes:
        tags = tags_for_node(node)
        if len(tags) > 0:
            title = etree.Element('title')
            title.text = ' '.join(tags)
            node.append(title)


def visualize_svg_annotations(svg_root):
    nodes = list(nodes_with_tags(svg_root))
    style_block = etree.Element('style')
    style_block.text = style_block_for_tags(tags_for_nodes(nodes))
    svg_root.insert(0, style_block)
    add_title_to_nodes(nodes)
    return svg_root
