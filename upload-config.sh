#!/bin/bash

source prepare-shell.sh

gsutil cp "${LOCAL_CONFIG_PATH}/${COLOR_MAP_FILENAME}" "$GCS_CONFIG_PATH/"
