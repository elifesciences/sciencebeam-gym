FROM python:3.7.10-buster AS base

RUN apt-get update \
  && apt-get install --assume-yes \
    poppler-utils \
  && rm -rf /var/lib/apt/lists/*

ENV PROJECT_FOLDER=/srv/sciencebeam-gym
WORKDIR ${PROJECT_FOLDER}

COPY requirements.build.txt ./
RUN pip install --disable-pip-version-check --no-warn-script-location --user \
  -r requirements.build.txt

COPY requirements.prereq.txt requirements.txt ./
RUN pip install --disable-pip-version-check --no-warn-script-location --user \
  -r requirements.prereq.txt \
  -r requirements.txt

RUN python -m nltk.downloader punkt


FROM base AS runtime

COPY sciencebeam_gym ./sciencebeam_gym
COPY *.conf *.sh *.in *.txt *.py ./

COPY scripts ./scripts
ENV PATH ${PROJECT_FOLDER}/scripts:$PATH


FROM base AS dev

COPY requirements.dev.txt ./
RUN pip install --disable-pip-version-check --no-warn-script-location --user \
  -r requirements.dev.txt

COPY sciencebeam_gym ./sciencebeam_gym
COPY tests ./tests
COPY *.conf *.sh *.in *.txt *.py .pylintrc .flake8 pytest.ini ./
