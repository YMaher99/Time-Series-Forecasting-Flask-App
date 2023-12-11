FROM python:3.11

# USER app
ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
ADD . /code
RUN export MLFLOW_HOME=/code
RUN export MLFLOW_TRACKING_URI=sqlite:///code/mlflow.sqlite

