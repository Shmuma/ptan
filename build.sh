#!/usr/bin/env bash

set -e

export IMAGE_NAME=nvcr.io/mitnvda18/fsa-atari

sudo nvidia-docker build -t $IMAGE_NAME .
sudo nvidia-docker push $IMAGE_NAME:latest