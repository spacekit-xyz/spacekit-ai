# Build growformer release binary for Ubuntu 22.04 LTS (x86_64).
# Use on cloud VMs (e.g. 48 vCPU, 96 GB) for faster MNIST runs.
#
# Build:  docker build -t growformer .
# Run:    docker run --rm -v /path/to/data:/data growformer --mnist
#         (set MNIST_ROOT=/data if your MNIST .ubyte files are in /data)
#
# Extract binary to run natively on host:
#   docker build -t growformer .
#   docker create --name gf growformer
#   docker cp gf:/usr/local/bin/growformer ./growformer-linux
#   docker rm gf

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app
COPY . .

RUN cargo build --release

# Minimal runtime image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/growformer /usr/local/bin/growformer

# Default: run MNIST benchmark. Override with docker run ... growformer --phase3c etc.
ENTRYPOINT ["/usr/local/bin/growformer"]
CMD ["--mnist"]
