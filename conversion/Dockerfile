# Environment for the Event-SAMA-239K conversion pipeline.
# Base: NVIDIA NGC TensorFlow 23.05 (TensorFlow 2.12 with GPU support, Python 3.10, CUDA 12.1),
# available for x86-64 and ARM64. TensorFlow is only used by the FILM upsampler.
#
#   docker build -t event-sama .
#
# The repository itself is mounted at runtime (see README), so code, FILM weights
# and outputs stay on the host.

FROM nvcr.io/nvidia/tensorflow:23.05-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /workspace
