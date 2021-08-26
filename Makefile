DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)

VENV = venv
PIP = $(VENV)/bin/pip
PYTHON = $(VENV)/bin/python

DEV_RUN = $(DOCKER_COMPOSE) run --rm sciencebeam-gym-dev

NOT_SLOW_PYTEST_ARGS = -m 'not slow'

ARGS =
PORT = 8080


.PHONY: all build


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	python3 -m venv $(VENV)


dev-install:
	$(PIP) install -r requirements.build.txt
	$(PIP) install \
		-r requirements.prereq.txt \
		-r requirements.txt \
		-r requirements.dev.txt


dev-nltk-download-models:
	$(PYTHON) -m nltk.downloader punkt


dev-venv: venv-create dev-install dev-nltk-download-models


dev-flake8:
	$(PYTHON) -m flake8 sciencebeam_gym tests setup.py


dev-pylint:
	$(PYTHON) -m pylint sciencebeam_gym tests setup.py


dev-mypy:
	$(PYTHON) -m mypy --ignore-missing-imports sciencebeam_gym tests setup.py $(ARGS)


dev-lint: dev-flake8 dev-pylint dev-mypy


dev-pytest:
	$(PYTHON) -m pytest -p no:cacheprovider $(ARGS)


.dev-watch:
	$(PYTHON) -m pytest_watch -- -p no:cacheprovider -p no:warnings $(ARGS)


dev-watch-slow:
	@$(MAKE) .dev-watch


dev-watch:
	@$(MAKE) ARGS="$(ARGS) $(NOT_SLOW_PYTEST_ARGS)" .dev-watch


dev-test: dev-lint dev-pytest


build:
	$(DOCKER_COMPOSE) build sciencebeam-gym


build-dev:
	$(DOCKER_COMPOSE) build sciencebeam-gym-base-dev sciencebeam-gym-dev


pylint:
	$(DEV_RUN) pylint sciencebeam_gym tests setup.py


flake8:
	$(DEV_RUN) flake8 sciencebeam_gym tests setup.py


mypy:
	$(DEV_RUN) mypy --ignore-missing-imports sciencebeam_gym tests setup.py


lint: \
	flake8 \
	pylint \
	mypy


pytest: build-dev
	$(DEV_RUN) pytest $(ARGS)


pytest-not-slow: build-dev
	@$(MAKE) ARGS="$(ARGS) $(NOT_SLOW_PYTEST_ARGS)" pytest


test: \
	lint \
	pytest \


.require-AUTOCUT_MODEL_PATH:
	@if [ -z "$(AUTOCUT_MODEL_PATH)" ]; then \
		echo "AUTOCUT_MODEL_PATH required"; \
		exit 1; \
	fi


shell-dev:
	$(DOCKER_COMPOSE) run --rm sciencebeam-gym bash


autocut-start: .require-AUTOCUT_MODEL_PATH build
	$(DOCKER_COMPOSE) run --rm \
	-v "$(AUTOCUT_MODEL_PATH):/tmp/model.pkl" \
	-e "AUTOCUT_MODEL_PATH=/tmp/model.pkl" \
	-p $(PORT):8080 \
	sciencebeam-gym \
	start-autocut.sh


autocut-start-cloud: .require-AUTOCUT_MODEL_PATH build
	$(DOCKER_COMPOSE) run --rm \
	-v $$HOME/.config/gcloud:/root/.config/gcloud \
	-e "AUTOCUT_MODEL_PATH=$(AUTOCUT_MODEL_PATH)" \
	-p $(PORT):8080 \
	sciencebeam-gym \
	start-autocut.sh


ci-build-and-test:
	make DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" \
		build build-dev mypy test


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
