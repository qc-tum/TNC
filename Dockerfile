FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update -qq && \
    apt-get install -qq openssh-client && \
    apt-get install -qq curl && \
    curl https://sh.rustup.rs > rustup-init.sh && \
    chmod +x ./rustup-init.sh && \
    ./rustup-init.sh -y --default-toolchain none  --no-modify-path && \
    rustup toolchain install stable --profile minimal && \
    rustup component add clippy rustfmt && \
    apt-get install -qq build-essential clang cmake gfortran libopenblas-dev libssl-dev pkg-config

