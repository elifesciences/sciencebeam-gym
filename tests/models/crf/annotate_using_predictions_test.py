from mock import MagicMock

import pytest

from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimplePage,
    SimpleLine,
    SimpleToken
)

from sciencebeam_gym.models.text.feature_extractor import (
    structured_document_to_token_props,
    token_props_list_to_features,
    NONE_TAG
)

from sciencebeam_gym.utils.bounding_box import (
    BoundingBox
)

from sciencebeam_gym.models.text.crf.annotate_using_predictions import (
    annotate_structured_document_using_predictions,
    predict_and_annotate_structured_document,
    CRF_TAG_SCOPE
)


TAG_1 = 'tag1'

TOKEN_TEXT_1 = 'token 1'
TOKEN_TEXT_2 = 'token 2'

BOUNDING_BOX = BoundingBox(0, 0, 10, 10)


class TestAnnotateStructuredDocumentUsingPredictions(object):
    def test_should_not_fail_with_empty_document(self):
        structured_document = SimpleStructuredDocument()
        annotate_structured_document_using_predictions(
            structured_document,
            []
        )

    def test_should_tag_single_token_using_prediction(self):
        token_1 = SimpleToken(TOKEN_TEXT_1)
        structured_document = SimpleStructuredDocument(lines=[SimpleLine([token_1])])
        annotate_structured_document_using_predictions(
            structured_document,
            [TAG_1]
        )
        assert structured_document.get_tag(token_1, scope=CRF_TAG_SCOPE) == TAG_1

    def test_should_not_tag_using_none_tag(self):
        token_1 = SimpleToken(TOKEN_TEXT_1)
        structured_document = SimpleStructuredDocument(lines=[SimpleLine([token_1])])
        annotate_structured_document_using_predictions(
            structured_document,
            [NONE_TAG]
        )
        assert structured_document.get_tag(token_1, scope=CRF_TAG_SCOPE) is None

    def test_should_tag_single_token_using_prediction_and_check_token_props(self):
        token_1 = SimpleToken(TOKEN_TEXT_1, bounding_box=BOUNDING_BOX)
        structured_document = SimpleStructuredDocument(SimplePage(
            lines=[SimpleLine([token_1])],
            bounding_box=BOUNDING_BOX
        ))
        token_props_list = structured_document_to_token_props(structured_document)
        annotate_structured_document_using_predictions(
            structured_document,
            [TAG_1],
            token_props_list
        )
        assert structured_document.get_tag(token_1, scope=CRF_TAG_SCOPE) == TAG_1

    def test_should_raise_error_if_token_props_do_not_match(self):
        token_1 = SimpleToken(TOKEN_TEXT_1, bounding_box=BOUNDING_BOX)
        structured_document = SimpleStructuredDocument(SimplePage(
            lines=[SimpleLine([token_1])],
            bounding_box=BOUNDING_BOX
        ))
        token_props_list = list(structured_document_to_token_props(structured_document))
        token_props_list[0]['text'] = TOKEN_TEXT_2
        with pytest.raises(AssertionError):
            annotate_structured_document_using_predictions(
                structured_document,
                [TAG_1],
                token_props_list
            )


class TestPredictAndAnnotateStructuredDocument(object):
    def test_should_predict_and_annotate_single_token(self):
        token_1 = SimpleToken(TOKEN_TEXT_1, bounding_box=BOUNDING_BOX)
        structured_document = SimpleStructuredDocument(SimplePage(
            lines=[SimpleLine([token_1])],
            bounding_box=BOUNDING_BOX
        ))
        model = MagicMock()
        model.predict.return_value = [[TAG_1]]
        token_props = list(structured_document_to_token_props(structured_document))
        X = [token_props_list_to_features(token_props)]
        predict_and_annotate_structured_document(
            structured_document,
            model
        )
        assert structured_document.get_tag(token_1, scope=CRF_TAG_SCOPE) == TAG_1
        model.predict.assert_called_with(X)
