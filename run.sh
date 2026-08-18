#!/usr/bin/env sh
set -eu

docker build --tag find-circle .
docker run --rm -it -v "$(pwd)/artifacts:/app/artifacts" find-circle "$@"
