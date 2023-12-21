FROM ubuntu:23.04
ENV DEBIAN_FRONTEND="noninteractive" \
    PATH="/root/.cargo/bin:${PATH}"
COPY rust-toolchain.toml .
RUN apt-get update && \
    # 1. Install dependencies for the toolchain
    apt-get install -y openssh-client jq && \
    # 2. Install rust
    apt-get install -y build-essential curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --no-modify-path && \
    # 3. Trigger installation of rust
    rustup show && \
    # 4. Install dependencies for fetching dependencies
    apt-get install -y git && \
    # 5. Install dependencies for the project
    # 6. Clear intermediate files
    apt-get clean
COPY Cargo.toml Cargo.lock tmp/
ARG SSH_PRIVATE_KEY
RUN --mount=type=ssh \
    # 1. Install dependency crates
    # 1.1 Create a dummy project
    cargo new dummy && \
    mv -f tmp/Cargo.toml tmp/Cargo.lock rust-toolchain.toml dummy && \
    cd dummy && \
    # 1.2 Set up SSH access to Gitlab
    mkdir -p -m 0700 ~/.ssh && \
    ssh-keyscan -H gitlab.lrz.de >> ~/.ssh/known_hosts && \
    if [ -z ${SSH_PRIVATE_KEY} ]; then \
    # To use SSH in local builds, use `docker build --ssh default .`
    # (or `docker build --ssh default=${HOME}/.ssh/id_rsa .`} for a specific key)
    # If the key is password protected, you can use `ssh-add ${HOME}/.ssh/mykey` first
    echo "No SSH_PRIVATE_KEY provided, using SSH from the host"; \
    else \
    # The SSH_PRIVATE_KEY build argument must be a base64-encoded private key
    # See https://stackoverflow.com/a/38570269
    # and https://www.programonaut.com/how-to-mask-an-ssh-private-key-in-gitlab-ci/
    echo "Using SSH_PRIVATE_KEY provided"; \
    eval $(ssh-agent -s); \
    echo "$SSH_PRIVATE_KEY" | base64 -d | tr -d '\r' | ssh-add -; \
    fi && \
    # 1.3 Install dependencies
    cargo fetch && \
    # 2. Clean intermediate files
    cd .. && \
    rm -rf dummy
