FROM python:3.12-slim-bookworm

RUN apt-get update

ARG MS_VERSION="latest"

RUN pip install --no-cache-dir macrosynergy==${MS_VERSION}