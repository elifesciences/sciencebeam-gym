#!/bin/bash

source prepare-shell.sh

if [ $USE_CLOUD == true ]; then
  PORT=6007
else
  PORT=6006
fi

tensorboard --logdir="${TRAIN_MODEL_PATH}/" --port=$PORT
