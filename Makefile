DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)


PYTEST_ARGS =


.PHONY: all build


dev-venv:
	if [ ! -e "venv/bin/python2.7" ]; then \
		rm -rf venv || true; \
		virtualenv -p python2.7 venv; \
	fi

	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements.prereq.txt
	venv/bin/pip install -r requirements.dev.txt
	venv/bin/python -m nltk.downloader punkt


build:
	$(DOCKER_COMPOSE) build sciencebeam-gym


build-dev:
	$(DOCKER_COMPOSE) build sciencebeam-gym-base-dev sciencebeam-gym-dev


test: build-dev
	$(DOCKER_COMPOSE) run --rm sciencebeam-gym-dev ./project_tests.sh


pytest: build-dev
	$(DOCKER_COMPOSE) run --rm sciencebeam-gym-dev pytest $(PYTEST_ARGS)


pytest-not-slow: build-dev
	$(DOCKER_COMPOSE) run --rm sciencebeam-gym-dev pytest -m 'not slow' $(PYTEST_ARGS)


autocut-start: build
	$(DOCKER_COMPOSE) run --rm \
	-v "$(AUTOCUT_MODEL_PATH):/tmp/model.pkl" \
	-e "AUTOCUT_MODEL_PATH=/tmp/model.pkl" \
	-p 8080:8080 \
	sciencebeam-gym \
	gunicorn \
  	'sciencebeam_gym.models.text.crf.autocut_app:create_app()' \
		--timeout 10 --log-level debug --workers 1 --worker-class gevent \
		 --bind 0.0.0.0:8080


ci-build-and-test:
	make DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" build test


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
