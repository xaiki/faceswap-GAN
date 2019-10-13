FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3-jupyter
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq -y \
  && apt-get install -y python3-opencv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


