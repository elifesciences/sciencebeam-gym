#!/bin/bash

set -e

GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT:-10}
GUNICORN_LOG_LEVEL=${GUNICORN_LOG_LEVEL:-info}
GUNICORN_WORKERS=${GUNICORN_WORKERS:-1}
GUNICORN_WORKER_CLASS=${GUNICORN_WORKER_CLASS:-gevent}
GUNICORN_HOST=${GUNICORN_HOST:-0.0.0.0}
GUNICORN_PORT=${GUNICORN_PORT:-8080}


if [ -z "${APP_NAME}" ]; then
    echo "APP_NAME required"
    exit 1
fi


APP_FACTORY_SUFFIX=':create_app()'


CMD="gunicorn \
    "${APP_NAME}${APP_FACTORY_SUFFIX}" \
    --timeout "${GUNICORN_TIMEOUT}" \
    --log-level "${GUNICORN_LOG_LEVEL}" \
    --workers "${GUNICORN_WORKERS}" \
    --worker-class "${GUNICORN_WORKER_CLASS}" \
    --bind "${GUNICORN_HOST}:${GUNICORN_PORT}" \
    $@"

echo $CMD

exec $CMD
