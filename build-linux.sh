#!/bin/bash
# Build Growformer for Linux amd64 (e.g. Ubuntu 22.04)
# Run from macOS or Linux; uses Docker to produce a Linux binary.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
OUTPUT_DIR="${SCRIPT_DIR}/build"
BINARY_NAME="growformer"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

build_with_docker() {
    log_info "Building with Docker (linux/amd64)..."

    if ! command -v docker &>/dev/null; then
        log_error "Docker not found. Install Docker Desktop."
        exit 1
    fi
    if ! docker info &>/dev/null; then
        log_error "Docker is not running. Please start Docker."
        exit 1
    fi

    mkdir -p "${OUTPUT_DIR}"

    log_info "Building ${BINARY_NAME} for Linux amd64..."
    docker run --rm \
        --platform linux/amd64 \
        -v "${PROJECT_ROOT}:/workspace" \
        -w /workspace \
        rust:1.85-slim-bookworm \
        bash -c "
            set -e
            echo '[INFO] Installing build dependencies...'
            apt-get update -qq
            apt-get install -y -qq pkg-config libssl-dev

            echo '[INFO] Building ${BINARY_NAME}...'
            cargo build --release

            echo '[SUCCESS] Build complete!'
        "

    copy_binary
}

copy_binary() {
    log_info "Copying binary to output directory..."
    mkdir -p "${OUTPUT_DIR}"

    local bin="${PROJECT_ROOT}/target/release/${BINARY_NAME}"
    if [ -f "${bin}" ]; then
        cp "${bin}" "${OUTPUT_DIR}/${BINARY_NAME}"
        chmod +x "${OUTPUT_DIR}/${BINARY_NAME}"
        log_success "Binary: ${OUTPUT_DIR}/${BINARY_NAME}"
        ls -la "${OUTPUT_DIR}/${BINARY_NAME}"
    else
        log_error "Binary not found at ${bin}"
        exit 1
    fi
}

case "${1:-docker}" in
    docker)
        build_with_docker
        ;;
    *)
        echo "Usage: $0 {docker}"
        echo "  docker - Use Docker to build Linux amd64 binary (recommended)"
        exit 1
        ;;
esac
