from functools import partial

import pytest

from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimpleLine,
    SimpleToken,
    merge_token_tag
)


TEXT_1 = 'text 1'
TEXT_2 = 'text 2'

TAG_1 = 'tag1'
TAG_2 = 'tag2'

SCOPE_1 = 'scope1'


class TestAbstractStructuredDocumentMergeWith(object):
    def test_should_merge_single_token_and_add_prefix(self):
        merged_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1, tag=TAG_1)
        ])])
        other_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1, tag=TAG_2)
        ])])
        merged_structured_document.merge_with(
            other_structured_document,
            partial(
                merge_token_tag,
                target_scope=SCOPE_1
            )
        )
        merged_tokens = list(merged_structured_document.iter_all_tokens())
        assert (
            [merged_structured_document.get_text(t) for t in merged_tokens] ==
            [TEXT_1]
        )
        assert (
            [merged_structured_document.get_tag(t) for t in merged_tokens] ==
            [TAG_1]
        )
        assert (
            [merged_structured_document.get_tag(t, scope=SCOPE_1) for t in merged_tokens] ==
            [TAG_2]
        )

    def test_should_not_fail_with_absent_tags(self):
        merged_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1, tag=TAG_1)
        ])])
        other_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1)
        ])])
        merged_structured_document.merge_with(
            other_structured_document,
            partial(
                merge_token_tag,
                target_scope=SCOPE_1
            )
        )
        merged_tokens = list(merged_structured_document.iter_all_tokens())
        assert (
            [merged_structured_document.get_tag(t, scope=SCOPE_1) for t in merged_tokens] ==
            [None]
        )

    def test_should_not_override_with_empty_tags(self):
        merged_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1, tag=TAG_1)
        ])])
        other_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1)
        ])])
        merged_structured_document.merge_with(
            other_structured_document,
            partial(
                merge_token_tag
            )
        )
        merged_tokens = list(merged_structured_document.iter_all_tokens())
        assert (
            [merged_structured_document.get_tag(t) for t in merged_tokens] ==
            [TAG_1]
        )

    def test_should_raise_assertion_error_if_tokens_mismatch(self):
        merged_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_1, tag=TAG_1)
        ])])
        other_structured_document = SimpleStructuredDocument(lines=[SimpleLine([
            SimpleToken(TEXT_2, tag=TAG_2)
        ])])
        with pytest.raises(AssertionError):
            merged_structured_document.merge_with(
                other_structured_document,
                partial(
                    merge_token_tag,
                    target_scope=SCOPE_1
                )
            )


class TestSimpleStructuredDocument(object):
    def test_should_set_tag_without_scope(self):
        token = SimpleToken(TEXT_1)
        doc = SimpleStructuredDocument(lines=[SimpleLine([token])])
        doc.set_tag(token, TAG_1)
        assert doc.get_tag(token) == TAG_1

    def test_should_set_tag_with_scope(self):
        token = SimpleToken(TEXT_1)
        doc = SimpleStructuredDocument(lines=[SimpleLine([token])])
        doc.set_tag(token, TAG_1, scope=SCOPE_1)
        assert doc.get_tag(token, scope=SCOPE_1) == TAG_1
        assert doc.get_tag(token) is None

    def test_should_return_all_tag_by_scope(self):
        token = SimpleToken(TEXT_1)
        doc = SimpleStructuredDocument(lines=[SimpleLine([token])])
        doc.set_tag(token, TAG_1)
        doc.set_tag(token, TAG_2, scope=SCOPE_1)
        assert doc.get_tag(token) == TAG_1
        assert doc.get_tag(token, scope=SCOPE_1) == TAG_2
        assert doc.get_tag_by_scope(token) == {None: TAG_1, SCOPE_1: TAG_2}
