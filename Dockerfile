FROM python:2.7.14-stretch

ENV PROJECT_FOLDER=/srv/sciencebeam-gym

ENV VENV=${PROJECT_FOLDER}/venv
RUN virtualenv ${VENV}
ENV PYTHONUSERBASE=${VENV} PATH=${VENV}/bin:$PATH

WORKDIR ${PROJECT_FOLDER}

COPY requirements.prereq.txt ${PROJECT_FOLDER}/
RUN venv/bin/pip install -r requirements.prereq.txt

COPY requirements.txt ${PROJECT_FOLDER}/
RUN venv/bin/pip install -r requirements.txt

ARG install_dev
COPY requirements.dev.txt ./
RUN if [ "${install_dev}" = "y" ]; then pip install -r requirements.dev.txt; fi

COPY sciencebeam_gym ${PROJECT_FOLDER}/sciencebeam_gym
COPY *.conf *.sh *.in *.txt *.py ${PROJECT_FOLDER}/
