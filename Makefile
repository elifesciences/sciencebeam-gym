DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)


dev-venv:
	if [ ! -e "venv/bin/python2.7" ]; then \
		rm -rf venv || true; \
		virtualenv -p python2.7 venv; \
	fi

	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements.prereq.txt
	venv/bin/pip install -r requirements.dev.txt


build-dev:
	$(DOCKER_COMPOSE) build sciencebeam-gym-base-dev sciencebeam-gym-dev


test: build-dev
	$(DOCKER_COMPOSE) run --rm sciencebeam-gym-dev ./project_tests.sh


ci-build-and-test:
	make DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" test


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
