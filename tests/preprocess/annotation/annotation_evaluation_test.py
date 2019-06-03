from __future__ import division

from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimpleLine,
    SimpleToken,
    B_TAG_PREFIX,
    I_TAG_PREFIX
)

from sciencebeam_gym.preprocess.annotation.annotation_evaluation import (
    evaluate_document_by_page
)


TAG1 = 'tag1'


class TestEvaluateDocumentByPage(object):
    def test_should_return_ratio_and_count_of_tagged_tokens(self):
        tagged_tokens = [
            SimpleToken('this'),
            SimpleToken('is'),
            SimpleToken('tagged')
        ]
        not_tagged_tokens = [
            SimpleToken('this'),
            SimpleToken('isn\'t')
        ]
        doc = SimpleStructuredDocument(lines=[SimpleLine(
            tagged_tokens + not_tagged_tokens
        )])
        for token in tagged_tokens:
            doc.set_tag(token, TAG1)
        num_total = len(tagged_tokens) + len(not_tagged_tokens)
        results = evaluate_document_by_page(doc)
        assert results == [{
            'count': {
                TAG1: len(tagged_tokens),
                None: len(not_tagged_tokens)
            },
            'percentage': {
                TAG1: len(tagged_tokens) / num_total,
                None: len(not_tagged_tokens) / num_total
            }
        }]

    def test_should_strip_prefix(self):
        tagged_tokens = [
            SimpleToken('this', tag=TAG1, tag_prefix=B_TAG_PREFIX),
            SimpleToken('is', tag=TAG1, tag_prefix=I_TAG_PREFIX),
            SimpleToken('tagged', tag=TAG1, tag_prefix=I_TAG_PREFIX)
        ]
        doc = SimpleStructuredDocument(lines=[SimpleLine(
            tagged_tokens
        )])
        results = evaluate_document_by_page(doc)
        assert set(results[0]['count'].keys()) == {TAG1}
