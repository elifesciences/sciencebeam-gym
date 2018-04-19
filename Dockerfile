FROM python:2.7.14-stretch
ENV PROJECT_HOME=/srv/sciencebeam-gym

WORKDIR ${PROJECT_HOME}
RUN virtualenv venv

COPY requirements.prereq.txt ${PROJECT_HOME}/
RUN venv/bin/pip install -r requirements.prereq.txt

COPY requirements.txt ${PROJECT_HOME}/
RUN venv/bin/pip install -r requirements.txt

COPY sciencebeam_gym ${PROJECT_HOME}/sciencebeam_gym
COPY *.conf *.sh *.in *.txt *.py ${PROJECT_HOME}/
