import logging

from sciencebeam_gym.structured_document import (
    B_TAG_PREFIX
)


def get_logger():
    return logging.getLogger(__name__)


class ExtractedItem(object):
    def __init__(self, tag, text, tokens=None, tag_prefix=None, sub_items=None):
        self.tag = tag
        self.tag_prefix = tag_prefix
        self.text = text
        self.tokens = tokens or []
        self.sub_items = sub_items or []

    def extend(self, other_item):
        return ExtractedItem(
            self.tag,
            self.text + '\n' + other_item.text,
            tokens=self.tokens + other_item.tokens,
            tag_prefix=self.tag_prefix,
            sub_items=self.sub_items + other_item.sub_items
        )


def get_lines(structured_document):
    for page in structured_document.get_pages():
        for line in structured_document.get_lines_of_page(page):
            yield line


def extract_from_annotated_tokens(structured_document, tokens, tag_scope=None, level=None):
    previous_tokens = []
    previous_tag = None
    previous_tag_prefix = None
    for token in tokens:
        tag_prefix, tag = structured_document.get_tag_prefix_and_value(
            token, scope=tag_scope, level=level
        )
        if not previous_tokens:
            previous_tokens = [token]
            previous_tag = tag
            previous_tag_prefix = tag_prefix
        elif tag == previous_tag and tag_prefix != B_TAG_PREFIX:
            previous_tokens.append(token)
        else:
            yield ExtractedItem(
                previous_tag,
                ' '.join(structured_document.get_text(t) for t in previous_tokens),
                tokens=previous_tokens,
                tag_prefix=previous_tag_prefix
            )
            previous_tokens = [token]
            previous_tag = tag
            previous_tag_prefix = tag_prefix
    if previous_tokens:
        yield ExtractedItem(
            previous_tag,
            ' '.join(structured_document.get_text(t) for t in previous_tokens),
            tokens=previous_tokens,
            tag_prefix=previous_tag_prefix
        )


def with_sub_items(structured_document, extracted_item, tag_scope=None):
    return ExtractedItem(
        extracted_item.tag,
        extracted_item.text,
        tokens=extracted_item.tokens,
        tag_prefix=extracted_item.tag_prefix,
        sub_items=list(extract_from_annotated_tokens(
            structured_document, extracted_item.tokens,
            tag_scope=tag_scope, level=2
        ))
    )


def extract_from_annotated_lines(structured_document, lines, tag_scope=None):
    previous_item = None
    for line in lines:
        tokens = structured_document.get_tokens_of_line(line)
        for item in extract_from_annotated_tokens(structured_document, tokens, tag_scope=tag_scope):
            if previous_item is not None:
                if previous_item.tag == item.tag and item.tag_prefix != B_TAG_PREFIX:
                    previous_item = previous_item.extend(item)
                else:
                    yield with_sub_items(structured_document, previous_item, tag_scope=tag_scope)
                    previous_item = item
            else:
                previous_item = item
    if previous_item is not None:
        yield with_sub_items(structured_document, previous_item, tag_scope=tag_scope)


def extract_from_annotated_document(structured_document, tag_scope=None):
    return extract_from_annotated_lines(
        structured_document, get_lines(structured_document), tag_scope=tag_scope
    )
