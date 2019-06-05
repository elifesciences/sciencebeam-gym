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
            value = request.get_data()
        else:
            value = request.args.get('value')
        LOGGER.debug('value: %s', value)
        output_value = model.predict([value])[0]
        return Response(output_value)

    return app


def main():
    debug_enabled = False
    if os.environ.get('AUTOCUT_DEBUG') == '1':
        logging.root.setLevel('DEBUG')
        debug_enabled = True
    create_app().run(debug=debug_enabled)


if __name__ == "__main__":
    logging.basicConfig(level='INFO')

    main()
