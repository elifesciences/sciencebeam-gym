# Experimental Pipeline

This pipeline is currently under development. It uses the CRF or computer vision model trained by
[ScienceBeam Gym](https://github.com/elifesciences/sciencebeam-gym).

What you need before you can proceed:

- At least one of:
  - Path to [CRF model](https://github.com/elifesciences/sciencebeam-gym#training-crf-model)
  - Path to [exported computer vision model](https://github.com/elifesciences/sciencebeam-gym#export-inference-model)
  - `--use-grobid` option
- PDF files, as file list csv/tsv or glob pattern

To use the CRF model together with the CV model, the CRF model will have to be trained with the CV predictions.

The following command will process files locally:

```bash
python -m sciencebeam_gym.convert.conversion_pipeline \
  --data-path=./data \
  --pdf-file-list=./data/file-list-validation.tsv \
  --crf-model=path/to/crf-model.pkl \
  --cv-model-export-dir=./my-model/export \
  --output-path=./data-results \
  --pages=1 \
  --limit=100
```

The following command would process the first 100 files in the cloud using Dataflow:

```bash
python -m sciencebeam_gym.convert.conversion_pipeline \
  --data-path=gs://my-bucket/data \
  --pdf-file-list=gs://my-bucket/data/file-list-validation.tsv \
  --crf-model=path/to/crf-model.pkl \
  --cv-model-export-dir=gs://mybucket/my-model/export \
  --output-path=gs://my-bucket/data-results \
  --pages=1 \
  --limit=100 \
  --cloud
```

You can also enable the post processing of the extracted authors and affiliations using Grobid by adding `--use-grobid`. In that case Grobid will be started automatically. To use an existing version also add `--grobid-url=<api url>` with the url to the Grobid API. If the `--use-grobid` option is used without a CRF or CV model, then it will use Grobid to translate the PDF to XML.

Note: using Grobid as part of this pipeline is considered deprecated and will likely be removed.

For a full list of parameters:

```bash
python -m sciencebeam.examples.conversion_pipeline --help
```
