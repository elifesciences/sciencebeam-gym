FROM python:3.7.10-buster AS runtime

RUN apt-get update \
  && apt-get install poppler-utils --assume-yes \
  && rm -rf /var/lib/apt/lists/*

ENV PROJECT_FOLDER=/srv/sciencebeam-gym

ENV VENV=${PROJECT_FOLDER}/venv
RUN python3 -m venv ${VENV}
ENV PYTHONUSERBASE=${VENV} PATH=${VENV}/bin:$PATH

WORKDIR ${PROJECT_FOLDER}

COPY requirements.build.txt ${PROJECT_FOLDER}/
RUN venv/bin/pip install -r requirements.build.txt

COPY requirements.prereq.txt ${PROJECT_FOLDER}/
RUN venv/bin/pip install -r requirements.prereq.txt

COPY requirements.txt ${PROJECT_FOLDER}/
RUN venv/bin/pip install -r requirements.txt

RUN python -m nltk.downloader punkt

ARG install_dev
COPY requirements.dev.txt ./
RUN if [ "${install_dev}" = "y" ]; then pip install -r requirements.dev.txt; fi

COPY sciencebeam_gym ${PROJECT_FOLDER}/sciencebeam_gym
COPY *.conf *.sh *.in *.txt *.py ${PROJECT_FOLDER}/

COPY scripts ${PROJECT_FOLDER}/scripts
ENV PATH ${PROJECT_FOLDER}/scripts:$PATH


FROM runtime AS dev

COPY tests ${PROJECT_FOLDER}/tests
COPY .pylintrc .flake8 pytest.ini ${PROJECT_FOLDER}/
