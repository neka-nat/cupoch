
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

WORKDIR /work/cupoch

RUN apt-get update && apt-get install -y --no-install-recommends \
         curl \
         build-essential \
         libxinerama-dev \
         libxcursor-dev \
         libglu1-mesa-dev \
         xorg-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY . .

RUN cd src/python \
    poetry config virtualenvs.create false \
    && poetry run pip install -U pip \
    && poetry install

RUN mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_PNG=ON -DBUILD_JSONCPP=ON \
    && make install-pip-package
