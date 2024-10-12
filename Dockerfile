# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter

FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /code

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=ultron
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# System deps:
RUN apt-get install curl ffmpeg libsm6 libxext6  -y
# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
ENV PATH="/root/.local/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

COPY . /code/
# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["--ip=0.0.0.0","--allow-root","--no-browser"]

