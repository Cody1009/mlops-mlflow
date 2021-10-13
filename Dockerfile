FROM python:3.7-slim-buster

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y  git curl vim

RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pipenv awscli


COPY . /var/lib/mlflow

WORKDIR /var/lib/mlflow
RUN pipenv install --system

ENV MLFLOW_BASE_PATH /var/lib/mlflow
RUN chmod 755 startup.sh


CMD ["/bin/bash", "startup.sh"]
