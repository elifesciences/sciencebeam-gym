#!/bin/bash

set -e

export APP_NAME=sciencebeam_gym.models.text.crf.autocut_app

exec "$(dirname $0)/start-app.sh" $@
