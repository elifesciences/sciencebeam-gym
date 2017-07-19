#!/bin/bash

source prepare-shell.sh

tensorboard --logdir="${TRAIN_MODEL_PATH}/"
