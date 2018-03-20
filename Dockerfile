FROM python:2.7.14-stretch
ARG tensorflow_version=1.4.0
ARG apache_beam_version=2.2.0
WORKDIR /srv/sciencebeam-gym
RUN virtualenv venv
RUN venv/bin/pip install tensorflow==${tensorflow_version}
RUN venv/bin/pip install apache-beam==${apache_beam_version}
COPY sciencebeam_gym /srv/sciencebeam-gym/sciencebeam_gym
COPY *.conf *.sh *.in *.txt *.py /srv/sciencebeam-gym/
RUN venv/bin/pip install -r requirements.txt
