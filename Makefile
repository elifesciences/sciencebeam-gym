DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)


PYTEST_ARGS =
PORT = 8080


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
	make DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" build test


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
