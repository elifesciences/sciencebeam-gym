DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)


build:
	$(DOCKER_COMPOSE) build


test: build
	$(DOCKER_COMPOSE) run --rm sciencebeam-gym ./project_tests.sh


ci-build-and-test:
	make DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" test


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
