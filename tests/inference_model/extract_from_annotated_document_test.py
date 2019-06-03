import logging

from sciencebeam_gym.structured_document import (
    SimpleToken,
    SimpleLine,
    SimpleStructuredDocument,
    B_TAG_PREFIX,
    I_TAG_PREFIX
)

from sciencebeam_gym.inference_model.extract_from_annotated_document import (
    extract_from_annotated_document
)


VALUE_1 = 'value1'
VALUE_2 = 'value2'
VALUE_3 = 'value3'
TEXT_1 = 'some text goes here'
TEXT_2 = 'another line another text'
TEXT_3 = 'more to come'
TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'

TAG_SCOPE_1 = 'tag_scope1'


def get_logger():
    return logging.getLogger(__name__)


def with_tag(x, tag):
    if isinstance(x, SimpleToken):
        x.set_tag(tag)
    elif isinstance(x, list):
        return [with_tag(y, tag) for y in x]
    elif isinstance(x, SimpleLine):
        return SimpleLine(with_tag(x.tokens, tag))
    return x


def to_token(token):
    return SimpleToken(token) if isinstance(token, str) else token


def to_tokens(tokens):
    if isinstance(tokens, str):
        tokens = tokens.split(' ')
    return [to_token(t) for t in tokens]


def to_line(tokens):
    return SimpleLine(to_tokens(tokens))


def annotated_tokens(tokens, tag):
    return with_tag(to_tokens(tokens), tag)


def annotated_line(tokens, tag):
    return with_tag(to_line(tokens), tag)


def _token_with_sub_tag(text, tag=None, tag_prefix=None, sub_tag=None, sub_tag_prefix=None):
    token = SimpleToken(text, tag=tag, tag_prefix=tag_prefix)
    if sub_tag:
        token.set_tag(sub_tag, prefix=sub_tag_prefix, level=2)
    return token


class TestExtractFromAnnotatedDocument(object):
    def test_should_not_fail_on_empty_document(self):
        structured_document = SimpleStructuredDocument()
        extract_from_annotated_document(structured_document)

    def test_should_extract_single_annotated_line(self):
        lines = [annotated_line(TEXT_1, TAG_1)]
        structured_document = SimpleStructuredDocument(lines=lines)
        result = [
            (x.tag, x.text)
            for x in
            extract_from_annotated_document(structured_document)
        ]
        assert result == [(TAG_1, TEXT_1)]

    def test_should_extract_from_different_tag_scope(self):
        lines = [SimpleLine([SimpleToken(TEXT_1, tag=TAG_1, tag_scope=TAG_SCOPE_1)])]
        structured_document = SimpleStructuredDocument(lines=lines)
        result = [
            (x.tag, x.text)
            for x in
            extract_from_annotated_document(structured_document, tag_scope=TAG_SCOPE_1)
        ]
        assert result == [(TAG_1, TEXT_1)]

    def test_should_extract_multiple_annotations_on_single_line(self):
        lines = [to_line(
            annotated_tokens(TEXT_1, TAG_1) +
            to_tokens(TEXT_2) +
            annotated_tokens(TEXT_3, TAG_3)
        )]
        structured_document = SimpleStructuredDocument(lines=lines)
        result = [
            (x.tag, x.text)
            for x in
            extract_from_annotated_document(structured_document)
        ]
        assert result == [
            (TAG_1, TEXT_1),
            (None, TEXT_2),
            (TAG_3, TEXT_3)
        ]

    def test_should_combine_multiple_lines(self):
        lines = [
            annotated_line(TEXT_1, TAG_1),
            annotated_line(TEXT_2, TAG_1)
        ]
        structured_document = SimpleStructuredDocument(lines=lines)
        result = [
            (x.tag, x.text)
            for x in
            extract_from_annotated_document(structured_document)
        ]
        get_logger().debug('result: %s', result)
        assert result == [(TAG_1, '\n'.join([TEXT_1, TEXT_2]))]

    def test_should_combine_multiple_lines_separated_by_other_tag(self):
        lines = [
            annotated_line(TEXT_1, TAG_1),
            annotated_line(TEXT_2, TAG_2),
            annotated_line(TEXT_3, TAG_2),
            annotated_line(TEXT_1, TAG_1),
            annotated_line(TEXT_2, TAG_2),
            annotated_line(TEXT_3, TAG_2)
        ]
        structured_document = SimpleStructuredDocument(lines=lines)
        result = [
            (x.tag, x.text)
            for x in
            extract_from_annotated_document(structured_document)
        ]
        get_logger().debug('result: %s', result)
        assert result == [
            (TAG_1, TEXT_1),
            (TAG_2, '\n'.join([TEXT_2, TEXT_3])),
            (TAG_1, TEXT_1),
            (TAG_2, '\n'.join([TEXT_2, TEXT_3]))
        ]

    def test_should_separate_items_based_on_tag_prefix(self):
        tokens = [
            SimpleToken(VALUE_1, tag=TAG_1, tag_prefix=B_TAG_PREFIX),
            SimpleToken(VALUE_2, tag=TAG_1, tag_prefix=I_TAG_PREFIX),
            SimpleToken(VALUE_3, tag=TAG_1, tag_prefix=I_TAG_PREFIX),
            SimpleToken(VALUE_1, tag=TAG_1, tag_prefix=B_TAG_PREFIX),
            SimpleToken(VALUE_2, tag=TAG_1, tag_prefix=I_TAG_PREFIX),
            SimpleToken(VALUE_3, tag=TAG_1, tag_prefix=I_TAG_PREFIX)
        ]
        structured_document = SimpleStructuredDocument(lines=[SimpleLine(tokens)])
        result = [
            (x.tag, x.text)
            for x in
            extract_from_annotated_document(structured_document)
        ]
        get_logger().debug('result: %s', result)
        assert result == [
            (TAG_1, ' '.join([VALUE_1, VALUE_2, VALUE_3])),
            (TAG_1, ' '.join([VALUE_1, VALUE_2, VALUE_3]))
        ]

    def test_should_extract_sub_tags_from_single_item(self):
        tokens = [
            _token_with_sub_tag(
                VALUE_1,
                tag=TAG_1, tag_prefix=B_TAG_PREFIX,
                sub_tag=TAG_2, sub_tag_prefix=B_TAG_PREFIX
            ),
            _token_with_sub_tag(
                VALUE_2,
                tag=TAG_1, tag_prefix=I_TAG_PREFIX,
                sub_tag=TAG_2, sub_tag_prefix=I_TAG_PREFIX
            ),
            _token_with_sub_tag(
                VALUE_3,
                tag=TAG_1, tag_prefix=I_TAG_PREFIX,
                sub_tag=TAG_3, sub_tag_prefix=B_TAG_PREFIX
            )
        ]
        structured_document = SimpleStructuredDocument(lines=[SimpleLine(tokens)])
        extracted_items = list(extract_from_annotated_document(structured_document))
        result = [
            (
                x.tag, x.text, [(sub.tag, sub.text) for sub in x.sub_items]
            ) for x in extracted_items
        ]
        get_logger().debug('result: %s', result)
        assert result == [
            (
                TAG_1, ' '.join([VALUE_1, VALUE_2, VALUE_3]),
                [
                    (TAG_2, ' '.join([VALUE_1, VALUE_2])),
                    (TAG_3, VALUE_3)
                ]
            )
        ]

    def test_should_extract_sub_tags_from_multiple_items(self):
        tokens = [
            _token_with_sub_tag(
                VALUE_1,
                tag=TAG_1, tag_prefix=B_TAG_PREFIX,
                sub_tag=TAG_2, sub_tag_prefix=B_TAG_PREFIX
            ),
            _token_with_sub_tag(
                VALUE_2,
                tag=TAG_1, tag_prefix=I_TAG_PREFIX,
                sub_tag=TAG_2, sub_tag_prefix=I_TAG_PREFIX
            ),
            _token_with_sub_tag(
                VALUE_3,
                tag=TAG_1, tag_prefix=I_TAG_PREFIX,
                sub_tag=TAG_3, sub_tag_prefix=B_TAG_PREFIX
            ),

            _token_with_sub_tag(
                VALUE_1,
                tag=TAG_1, tag_prefix=B_TAG_PREFIX,
                sub_tag=TAG_2, sub_tag_prefix=B_TAG_PREFIX
            ),
            _token_with_sub_tag(
                VALUE_2,
                tag=TAG_1, tag_prefix=I_TAG_PREFIX,
                sub_tag=TAG_3, sub_tag_prefix=B_TAG_PREFIX
            ),
            _token_with_sub_tag(
                VALUE_3,
                tag=TAG_1, tag_prefix=I_TAG_PREFIX,
                sub_tag=TAG_3, sub_tag_prefix=I_TAG_PREFIX
            )
        ]
        structured_document = SimpleStructuredDocument(lines=[SimpleLine(tokens)])
        extracted_items = list(extract_from_annotated_document(structured_document))
        result = [
            (
                x.tag, x.text, [(sub.tag, sub.text) for sub in x.sub_items]
            ) for x in extracted_items
        ]
        get_logger().debug('result: %s', result)
        assert result == [
            (
                TAG_1, ' '.join([VALUE_1, VALUE_2, VALUE_3]),
                [
                    (TAG_2, ' '.join([VALUE_1, VALUE_2])),
                    (TAG_3, VALUE_3)
                ]
            ),
            (
                TAG_1, ' '.join([VALUE_1, VALUE_2, VALUE_3]),
                [
                    (TAG_2, VALUE_1),
                    (TAG_3, ' '.join([VALUE_2, VALUE_3]))
                ]
            )
        ]
