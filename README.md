Science Beam - Gym
==================

This is where the [Science Beam](https://github.com/elifesciences/sciencebeam) model is trained.

Pre-requisites
--------------

* [Apache Beam](https://beam.apache.org/)
* [TensorFlow](https://www.tensorflow.org/) with google cloud support
* [gsutil](https://cloud.google.com/storage/docs/gsutil)

Local vs. Cloud
---------------

Almost all of the commands can be run locally or in the cloud. Simply add `--cloud` to the command to run it in the cloud. You will have to have [gsutil](https://cloud.google.com/storage/docs/gsutil) installed even when running locally.

Before running anything in the cloud, please run `upload-config.sh` to copy the required configuration to the cloud.

Configuration
-------------

The default configuration is in the [prepare-shell.sh](prepare-shell.sh) script. Some of the configuration can be overriden by adding a `.config` file which overrides some of the variables, e.g.:
```bash
#!/bin/bash

export TRAINING_SUFFIX=-gan-1-l1-100
export TRAINING_ARGS="--gan_weight=1 --l1_weight=100"
export USE_SEPARATE_CHANNELS=true
```

### Inspecting Configuration

By running `source prepare-shell.sh` the configuration can be inspected.

e.g. the following sequence of commands will print the data directory:
```bash
source prepare-shell.sh
echo $DATA_PATH
```

The following sections may refer to variables defined by that script.

Pipeline
--------

The TensorFlow training pipeline is illustrated in the following diagram:

![TensorFlow Training Pipeline](doc/pdf-xml-tf-training-pipeline.png)

The steps from the diagram are detailed below.

### Generate PNG

This step is currently not part of this repository (it will be made available in the future).

Instead you will need access to the annotated PNGs.

The data need to be made available in `$GCS_DATA_PATH` or `$LOCAL_DATA_PATH` depending on whether running it in the cloud.

Running `./upload-data` (optional) will copy files from `$LOCAL_DATA_PATH` to `$GCS_DATA_PATH`.

### Generate TFRecords

To make the training more efficient, it is recommended to use TFRecords for the training data.

The following script will resize the images from `$DATA_PATH` to the required size and generate TFRecords, which will be written to `$PREPROC_PATH`:

```bash
./preprocess.sh [--cloud]
```

### Train TF Model

Running the following command will train the model:

```bash
./train.sh [--cloud]
```

### Export Inference Model

This step is currently not implemented.

### TensorBoard

Run the TensorBoard with the correct path:

```bash
./tensorboard.sh [--cloud]
```
