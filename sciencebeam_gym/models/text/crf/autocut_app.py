import os
import pickle
import logging

from flask import Flask, Response, request

from apache_beam.io.filesystems import FileSystems


LOGGER = logging.getLogger(__name__)


def get_model_path():
    return os.environ['AUTOCUT_MODEL_PATH']


def load_model(file_path):
    with FileSystems.open(file_path) as fp:
        return pickle.load(fp)


def create_app():
    app = Flask(__name__)
    model = load_model(get_model_path())
    LOGGER.debug('loaded model: %s', model)

    @app.route('/api/autocut', methods=['GET', 'POST'])
    def _autocut():
        if request.method == 'POST':
            value = request.data
        else:
            value = request.args.get('value')
        LOGGER.debug('value: %s', value)
        output_value = model.predict([value])[0]
        return Response(output_value)

    return app


def main():
    create_app().run()


if __name__ == "__main__":
    main()
