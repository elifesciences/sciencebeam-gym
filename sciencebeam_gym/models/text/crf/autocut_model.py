import nltk

from sciencebeam_utils.utils.collection import extend_dict

from sciencebeam_alignment.align import (
    LocalSequenceMatcher,
    SimpleScoring
)

from .crfsuite_model import CrfSuiteModel


DEFAULT_SCORING = SimpleScoring(
    match_score=2,
    mismatch_score=-1,
    gap_score=-2
)

MATCH_LABEL = 'x'
OTHER_LABEL = '_'


def get_labels_match(expected, actual, match_label=MATCH_LABEL, other_label=OTHER_LABEL):
    if not actual:
        return ''
    if not expected:
        return '_' * len(actual)
    try:
        sm = LocalSequenceMatcher(a=expected.lower(), b=actual.lower(), scoring=DEFAULT_SCORING)
        matching_blocks = sm.get_matching_blocks()
        from_actual = min(b for _, b, size in matching_blocks if size)
        to_actual = max(b + size for _, b, size in matching_blocks if size)
        return (
            (other_label * from_actual)
            + (match_label * (to_actual - from_actual))
            + (other_label * (len(actual) - to_actual))
        )
    except IndexError as e:
        raise IndexError('%s: expected=[%s], actual=[%s]' % (e, expected, actual)) from e


def span_word_tokenize(txt):
    tokens = nltk.word_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset + len(token)
        offset += len(token)


def get_word_index_by_char_index_map(spans):
    return {
        index: word_index
        for word_index, (_, start, end) in enumerate(spans)
        for index in range(start, end)
    }


def get_span_words(spans):
    return [word for word, _, _ in spans]


def get_word_by_char_index_map(spans):
    spans = list(spans)
    words = get_span_words(spans)
    return {
        index: words[word_index]
        for index, word_index in get_word_index_by_char_index_map(spans).items()
    }


def get_char_features(prefix, ch):
    return {
        '%s.lower' % prefix: ch.lower(),
        '%s.isupper' % prefix: ch.isupper(),
        '%s.istitle' % prefix: ch.istitle(),
        '%s.isdigit' % prefix: ch.isdigit(),
    }


def get_word_features(prefix, word):
    return {
        '%s.lower' % prefix: word.lower(),
        '%s.isupper' % prefix: word.isupper(),
        '%s.istitle' % prefix: word.istitle(),
        '%s.isdigit' % prefix: word.isdigit(),
    }


def get_sentence_char_features(
        char_index,
        char_by_index_map,
        word_index_by_char_index_map,
        word_by_index_map):
    word_index = word_index_by_char_index_map.get(char_index, -10)
    d = extend_dict(
        {},
        get_char_features('char', char_by_index_map.get(char_index, '')),
        get_word_features('word', word_by_index_map.get(word_index, '')),
        {
            'char_index': char_index,
            'word_index': word_index,
            'bias': 1.0
        }
    )
    for i in range(1, 1 + 3):
        d.update(get_char_features('char[-%d]' % i, char_by_index_map.get(char_index - i, '')))
        d.update(get_char_features('char[+%d]' % i, char_by_index_map.get(char_index + i, '')))
        d.update(get_word_features('word[-%d]' % i, word_by_index_map.get(word_index - i, '')))
        d.update(get_word_features('word[+%d]' % i, word_by_index_map.get(word_index + i, '')))
    return d


def sentence_to_features(sentence):
    spans = list(span_word_tokenize(sentence))
    word_by_index_map = dict(enumerate(get_span_words(spans)))
    word_index_by_char_index_map = get_word_index_by_char_index_map(spans)
    char_by_index_map = dict(enumerate(sentence))
    return [
        get_sentence_char_features(
            char_index,
            char_by_index_map,
            word_index_by_char_index_map=word_index_by_char_index_map,
            word_by_index_map=word_by_index_map
        )
        for char_index in range(len(sentence))
    ]


def get_value_using_predicted_character_labels(
        source_value, character_labels,
        match_label=MATCH_LABEL, other_label=OTHER_LABEL):
    try:
        start = character_labels.index(match_label)
    except ValueError:
        return ''
    try:
        end = start + character_labels[start:].index(other_label, start)
    except ValueError:
        end = len(source_value)
    return source_value[start:end]


class AutocutModel(CrfSuiteModel):
    def _transform_x(self, X):
        return [sentence_to_features(item) for item in X]

    def _transform_y(self, y, X):
        return [
            get_labels_match(expected, actual)
            for expected, actual in zip(y, X)
        ]

    def _rev_transform_y(self, y_pred, X):
        return [
            get_value_using_predicted_character_labels(source_value, character_labels)
            for source_value, character_labels in zip(X, y_pred)
        ]

    def fit(self, X, y, X_dev=None, y_dev=None):
        super().fit(self._transform_x(X), self._transform_y(y, X=X))

    def predict(self, X):
        return self._rev_transform_y(
            super().predict(self._transform_x(X)),
            X=X
        )
