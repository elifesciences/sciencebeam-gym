FROM python:2.7.14-stretch
COPY sciencebeam_gym /srv/sciencebeam-gym/sciencebeam_gym
COPY *.conf *.sh *.in *.txt *.py /srv/sciencebeam-gym/
WORKDIR /srv/sciencebeam-gym
RUN virtualenv venv
RUN venv/bin/pip install -r requirements.txt
