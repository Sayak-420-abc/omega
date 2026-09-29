FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN bash scripts/fetch_assets.sh

RUN cmake -S . -B build -DBUILD_ONNXRUNTIME=ON -DBUILD_LIBTORCH=OFF -DBUILD_REST_API=ON \
    && cmake --build build --config Release -j

CMD ["./build/bin/highperf-ai-ml-inference", "--backend", "onnx", "--model", "models/squeezenet1.1.onnx", "--input", "assets/sample.jpg", "--topk", "5"]
