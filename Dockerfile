FROM ubuntu:23.04
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
ADD rust-toolchain.toml .
RUN apt-get update -qq && \
    apt-get install -qq openssh-client && \
    apt-get install -qq curl && \
    curl https://sh.rustup.rs > rustup-init.sh && \
    chmod +x ./rustup-init.sh && \
    ./rustup-init.sh -y --default-toolchain none  --no-modify-path && \
    rustup show && \
    apt-get install -qq build-essential clang cmake gfortran libopenblas-dev libssl-dev libhdf5-dev pkg-config libboost-all-dev
