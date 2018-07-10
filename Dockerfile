
FROM nvcr.io/nvidia/pytorch:18.06-py3
MAINTAINER Brandon Araki <araki@mit.edu>

# ubuntu packages
RUN apt-get update && apt-get install -y \
  git \
  wget \ 
  unzip \ 
  vim \
  libhdf5-serial-dev hdf5-tools \
  libmatio2 \
  ssh \
  xvfb \
  libav-tools \
  libavcodec-dev \
  python-opengl

RUN apt-get -y build-dep python-matplotlib
RUN apt-get update && apt-get upgrade -y

RUN pip install --upgrade pip

WORKDIR ptan
COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt

# get the latest pytorch
RUN pip uninstall torch -y
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl 

RUN pip install --extra-index-url=https://packages.nvidia.com/ngc/ngc-sdk/pypi/simple telemetry --upgrade # install telemetry for the NGC cloud

COPY fsa-atari/ .
RUN pip install -e gym-fsa-atari

COPY setup.py .
RUN python setup.py install

COPY . .