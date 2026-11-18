FROM python:3.12-slim-bookworm AS base

RUN apt-get update

ARG MS_VERSION

# develop: macrosynergy from the build context (a checkout of develop). The
# source is bind-mounted, not copied, so no source layer lands in the image.
FROM base AS develop
RUN --mount=type=bind,target=/src,rw pip install --no-cache-dir /src

# release: macrosynergy==MS_VERSION from PyPI. Keep it last: docker-ecr-push.yml
# builds the default (last) stage for a release.
FROM base AS release
RUN pip install --no-cache-dir macrosynergy==${MS_VERSION}
