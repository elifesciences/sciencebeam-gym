FROM python:2.7.14-stretch

ENV PROJECT_HOME=/srv/sciencebeam-gym

ENV VENV=${PROJECT_HOME}/venv
RUN virtualenv ${VENV}
ENV PYTHONUSERBASE=${VENV} PATH=${VENV}/bin:$PATH

WORKDIR ${PROJECT_HOME}

COPY requirements.prereq.txt ${PROJECT_HOME}/
RUN venv/bin/pip install -r requirements.prereq.txt

COPY requirements.txt ${PROJECT_HOME}/
RUN venv/bin/pip install -r requirements.txt

COPY sciencebeam_gym ${PROJECT_HOME}/sciencebeam_gym
COPY *.conf *.sh *.in *.txt *.py ${PROJECT_HOME}/

# tests
COPY .pylintrc .flake8 ${PROJECT_HOME}/
