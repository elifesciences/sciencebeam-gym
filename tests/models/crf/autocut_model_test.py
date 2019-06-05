import pickle


from sciencebeam_gym.models.text.crf.autocut_model import (
    AutocutModel
)


TITLE_1 = 'The scientific life of mice'
TITLE_2 = 'Cat and mouse'
TITLE_3 = 'The answer to everything'
TITLE_4 = 'A journey from PDF to XML'


def _fit_predict_model(X_train, y_train, X_test=None):
    model = AutocutModel()
    model.fit(X_train, y_train)
    pred_model = pickle.loads(pickle.dumps(model))
    return pred_model.predict(X_test or X_train)


class TestAutocutModel(object):
    def test_should_learn_to_remove_simple_prefix_on_train_data(self):
        X = ['Title: ' + TITLE_1, TITLE_2]
        y = [TITLE_1, TITLE_2]
        y_pred = _fit_predict_model(X, y)
        assert y_pred == y

    def test_should_learn_to_remove_simple_prefix(self):
        X_train = ['Title: ' + TITLE_1]
        y_train = [TITLE_1]
        X_test = ['Title: ' + TITLE_2]
        y_test = [TITLE_2]
        y_pred = _fit_predict_model(X_train, y_train, X_test)
        assert y_pred == y_test

    def test_should_learn_to_remove_extra_text(self):
        X_train = [TITLE_1 + ' Sub title: ' + TITLE_2]
        y_train = [TITLE_1]
        X_test = [TITLE_3 + ' Sub title:' + TITLE_4]
        y_test = [TITLE_3]
        y_pred = _fit_predict_model(X_train, y_train, X_test)
        assert y_pred == y_test

    def test_should_learn_to_remove_prefix_and_extra_text(self):
        X_train = ['Title: ' + TITLE_1 + ' Sub title: ' + TITLE_2]
        y_train = [TITLE_1]
        X_test = ['Title: ' + TITLE_3 + ' Sub title:' + TITLE_4]
        y_test = [TITLE_3]
        y_pred = _fit_predict_model(X_train, y_train, X_test)
        assert y_pred == y_test
