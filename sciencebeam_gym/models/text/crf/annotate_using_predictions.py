import argparse
import logging
import pickle
from itertools import repeat

from sciencebeam_gym.utils.tf import (
    FileIO
)

from sciencebeam_gym.models.text.feature_extractor import (
    structured_document_to_token_props,
    token_props_list_to_features,
    merge_with_cv_structured_document,
    NONE_TAG
)

from sciencebeam_gym.structured_document.structured_document_loader import (
    load_lxml_structured_document
)

from sciencebeam_gym.structured_document.structured_document_saver import (
    save_structured_document
)

CRF_TAG_SCOPE = 'crf'


def get_logger():
    return logging.getLogger(__name__)


def _iter_tokens(structured_document):
    for page in structured_document.get_pages():
        for line in structured_document.get_lines_of_page(page):
            for token in structured_document.get_tokens_of_line(line):
                yield token


def annotate_structured_document_using_predictions(
        structured_document, predictions, token_props_list=None,
        tag_scope=CRF_TAG_SCOPE):
    """
    Annotates the structured document using the predicted tags.

    Args:
      structured_document: the document that will be tagged
      predictions: list of predicted tags
      token_props_list: optional, used to verify that the correct token is being tagged
      tag_scope: tag scope to use when setting predicted tags
    """

    if token_props_list is None:
        token_props_list = repeat(None)
    for token, prediction, token_props in zip(
        _iter_tokens(structured_document),
        predictions, token_props_list
    ):

        if token_props:
            assert structured_document.get_text(token) == token_props['text']

        if prediction and prediction != NONE_TAG:
            structured_document.set_tag(token, prediction, scope=tag_scope)


def predict_and_annotate_structured_document(structured_document, model, tag_scope=CRF_TAG_SCOPE):
    token_props = list(structured_document_to_token_props(structured_document))
    x = token_props_list_to_features(token_props)
    y_pred = model.predict([x])[0]
    annotate_structured_document_using_predictions(
        structured_document, y_pred, token_props, tag_scope=tag_scope
    )
    return structured_document


def parse_args(argv=None):
    parser = argparse.ArgumentParser('Annotated LXML using CRF model')
    source = parser.add_argument_group('source')
    source.add_argument(
        '--lxml-path', type=str, required=False,
        help='path to lxml document'
    )

    cv_source = parser.add_argument_group('CV source')
    cv_source.add_argument(
        '--cv-lxml-path', type=str, required=False,
        help='path to lxml document with cv predicted tags'
    )

    parser.add_argument(
        '--crf-model', type=str, required=True,
        help='path to saved crf model'
    )

    parser.add_argument(
        '--output-path', type=str, required=True,
        help='output path to annotated document'
    )

    parser.add_argument(
        '--tag-scope', type=str, required=False,
        default=CRF_TAG_SCOPE,
        help='target tag scope for the predicted tags'
    )

    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='enable debug output'
    )

    return parser.parse_args(argv)


def load_crf_model(path):
    with FileIO(path, 'rb') as crf_model_f:
        return pickle.load(crf_model_f)


def main(argv=None):
    args = parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel('DEBUG')

    structured_document = load_lxml_structured_document(args.lxml_path)

    if args.cv_lxml_path:
        cv_structured_document = load_lxml_structured_document(args.cv_lxml_path)
        structured_document = merge_with_cv_structured_document(
            structured_document,
            cv_structured_document
        )

    model = load_crf_model(args.crf_model)

    predict_and_annotate_structured_document(
        structured_document,
        model,
        tag_scope=args.tag_scope
    )

    get_logger().info('writing result to: %s', args.output_path)
    save_structured_document(args.output_path, structured_document)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
