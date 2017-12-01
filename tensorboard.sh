#!/bin/bash

source prepare-shell.sh

if [ $USE_CLOUD == true ]; then
  PORT=6007
else
  PORT=6006
fi

LOGDIR=$TRAIN_MODEL_PATH/
echo "log dir: $LOGDIR"
tensorboard --logdir="${LOGDIR}" --port=$PORT
