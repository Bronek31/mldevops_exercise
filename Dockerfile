FROM python:3.12

RUN pip install poetry==1.8.4

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY . /app

RUN poetry install
