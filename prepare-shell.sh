#!/bin/bash

# Use this file by running:
# source prepare-shell.sh [--cloud]

export SUB_PROJECT_NAME="sciencebeam"
export MODEL_NAME="pix2pix"
export VERSION_NAME=v4
export TRAINING_SUFFIX=-default
export TRAINING_ARGS=""
export PROJECT=$(gcloud config list project --format "value(core.project)")
export LOCAL_PATH_ROOT="./.models"
export BUCKET="gs://${PROJECT}-ml"
export COLOR_MAP_FILENAME="color_map.conf"
export USE_SEPARATE_CHANNELS=true
export DATASET_SUFFIX=
export EVAL_SET_SIZE=10
export QUANTITATIVE_FOLDER_NAME=
export QUANTITATIVE_SET_SIZE=10

export USE_CLOUD=false

if [ "$1" == "--cloud" ]; then
  export USE_CLOUD=true
fi

export CONFIG_FILE='.config'
if [ -f "$CONFIG_FILE" ]; then
  source "${CONFIG_FILE}"
fi

# generate job id and save it
# TODO this should be done on-demand
export DEFAULT_JOB_ID="${SUB_PROJECT_NAME}_${USER}_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

export JOB_ID_FILE='.job-id'
if [ -f "$JOB_ID_FILE" ]; then
  export JOB_ID=`cat "${JOB_ID_FILE}"`
else
  export JOB_ID="${DEFAULT_JOB_ID}"
  echo -n "$JOB_ID" > "${JOB_ID_FILE}"
fi

# cloud paths
export GCS_SUB_PROJECT_PATH="${BUCKET}/${SUB_PROJECT_NAME}"
export GCS_PATH="${GCS_SUB_PROJECT_PATH}/${MODEL_NAME}/${VERSION_NAME}"
export GCS_DATA_PATH="${GCS_PATH}/data${DATASET_SUFFIX}"
export GCS_CONFIG_PATH="${GCS_PATH}/config"
export GCS_PREPROC_PATH="${GCS_PATH}/preproc${DATASET_SUFFIX}"
export GCS_TRAIN_MODEL_PATH="${GCS_PATH}${TRAINING_SUFFIX}/training"

# local paths
export LOCAL_MODEL_PATH="${LOCAL_PATH_ROOT}/${MODEL_NAME}/${VERSION_NAME}"
export LOCAL_DATA_PATH="${LOCAL_MODEL_PATH}/data${DATASET_SUFFIX}"
export LOCAL_CONFIG_PATH="."
export LOCAL_PREPROC_PATH="${LOCAL_MODEL_PATH}/preproc${DATASET_SUFFIX}"
export LOCAL_TRAIN_MODEL_PATH="${LOCAL_MODEL_PATH}${TRAINING_SUFFIX}/training"

echo "USE_CLOUD: $USE_CLOUD"

if [ $USE_CLOUD == true ]; then
  export DATA_PATH="${GCS_DATA_PATH}"
  export CONFIG_PATH="${GCS_CONFIG_PATH}"
  export PREPROC_PATH="${GCS_PREPROC_PATH}"
  export TRAIN_MODEL_PATH="${GCS_TRAIN_MODEL_PATH}"
else
  export DATA_PATH="${LOCAL_DATA_PATH}"
  export CONFIG_PATH="${LOCAL_CONFIG_PATH}"
  export PREPROC_PATH="${LOCAL_PREPROC_PATH}"
  export TRAIN_MODEL_PATH="${LOCAL_TRAIN_MODEL_PATH}"
fi
