#!/usr/bin/env bash
set -euo pipefail

# Docker-based precompilation script for nx_eigen
# This script builds native Docker containers for each target architecture
# and runs mix elixir_make.precompile in each one

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${ELIXIR_MAKE_CACHE_DIR:-${PROJECT_ROOT}/cache}"

# Define all supported targets and their Docker platforms
# Note: RISC-V support requires special Docker BuildKit setup and may not work on all systems
declare -A TARGETS=(
    ["x86_64-linux-gnu"]="linux/amd64"
    ["aarch64-linux-gnu"]="linux/arm64"
    ["aarch64-arduino-uno-q-linux-gnu"]="linux/arm64"
    ["x86_64-linux-musl"]="linux/amd64"
    ["aarch64-linux-musl"]="linux/arm64"
)

# Optional targets that may not be available on all systems
# Uncomment to enable RISC-V support (requires riscv64 emulation)
# TARGETS["riscv64-linux-gnu"]="linux/riscv64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}==>${NC} $1"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        log_error "Please install Docker: https://docs.docker.com/get-docker/"
        return 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        return 1
    fi

    log_info "Docker is available"
    return 0
}

# Build Docker image for a specific platform
build_docker_image_for_platform() {
    local platform="$1"
    local tag="$2"
    local target="$3"

    log_step "Building Docker image for platform: ${platform}"

    cd "${PROJECT_ROOT}"

    # Use Alpine-based Dockerfile for musl targets
    local dockerfile="Dockerfile.precompile"
    if [[ "$target" == *"-musl" ]]; then
        dockerfile="Dockerfile.precompile.alpine"
        log_info "Using Alpine Linux for musl target"
    fi

    if ! docker buildx build \
        --file "${dockerfile}" \
        --platform "${platform}" \
        --tag "${tag}" \
        --build-arg TARGETPLATFORM="${platform}" \
        --load \
        .; then
        log_error "Failed to build Docker image for ${platform}"
        return 1
    fi

    log_info "Docker image built successfully for ${platform}"
}


# Run precompilation for a specific target inside Docker
run_precompile_for_target() {
    local target="$1"
    local docker_image="$2"

    log_step "Running precompilation for target: ${target}"

    mkdir -p "${CACHE_DIR}"

    # Set target-specific environment variables for Arduino Uno Q
    local extra_env=""
    if [[ "$target" == "aarch64-arduino-uno-q-linux-gnu" ]]; then
        extra_env="-e TARGET_OS=arduino-uno-q-linux -e TARGET_ARCH=aarch64 -e TARGET_ABI=gnu"
    fi

    # Use appropriate shell based on target (Alpine uses /bin/sh)
    local shell_cmd="bash"
    if [[ "$target" == *"-musl" ]]; then
        shell_cmd="sh"
    fi

    if ! docker run --rm \
        -v "${PROJECT_ROOT}:/work" \
        -w /work \
        -e MIX_ENV=prod \
        -e ELIXIR_MAKE_CACHE_DIR=/work/cache \
        -e PRECOMPILE_TARGET="${target}" \
        ${extra_env} \
        "${docker_image}" \
        ${shell_cmd} -c "
          set -e
          echo 'Installing Hex and Rebar...'
          mix local.hex --force
          mix local.rebar --force

          echo 'Getting dependencies...'
          mix deps.get

          echo 'Verifying FFTW availability...'
          pkg-config --exists fftw3 fftw3f && echo 'FFTW found via pkg-config' || echo 'WARNING: FFTW not found!'
          pkg-config --libs fftw3 fftw3f || echo 'WARNING: Could not get FFTW libs!'

          echo 'Running precompilation for ${target}...'
          export NX_EIGEN_FFT_LIB=fftw
          export USE_CMAKE=1
          mix elixir_make.precompile

          echo 'Verifying linked libraries in built binary...'
          LATEST_TARBALL=\$(ls -t cache/*${target}*.tar.gz 2>/dev/null | head -n 1)
          if [ -n "\$LATEST_TARBALL" ]; then
            mkdir -p /tmp/verify
            tar -xzf "\$LATEST_TARBALL" -C /tmp/verify
            echo 'Binary dependencies:'
            readelf -d /tmp/verify/libnx_eigen.so 2>/dev/null | grep NEEDED || echo 'Could not read dependencies'
            rm -rf /tmp/verify
          else
            echo 'No tarball found for verification'
          fi

          echo 'Precompilation complete for ${target}!'
        "; then
        log_warn "Precompilation failed for ${target}"
        return 1
    fi

    log_info "Precompilation complete for ${target}"
}


# Main execution
main() {
    local specific_target="${1:-}"

    log_info "Starting Docker-based precompilation for all targets"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Cache directory: ${CACHE_DIR}"
    log_info ""

    # Check prerequisites
    check_docker || exit 1

    # Check if buildx is available
    if ! docker buildx version &> /dev/null; then
        log_error "Docker buildx is required for multi-platform builds"
        log_error "Please enable buildx: docker buildx create --use"
        exit 1
    fi

    local failed_targets=()
    local successful_targets=()

    # Build and run precompilation for each target
    for target in "${!TARGETS[@]}"; do
        # If specific target requested, skip others
        if [[ -n "$specific_target" && "$target" != "$specific_target" ]]; then
            continue
        fi

        platform="${TARGETS[$target]}"
        docker_image="nx_eigen_builder:${target}"

        log_step "Processing target: ${target} (${platform})"

        # Build Docker image for this platform
        if ! build_docker_image_for_platform "${platform}" "${docker_image}" "${target}"; then
            log_error "Failed to build image for ${target}"
            failed_targets+=("${target}")
            continue
        fi

        # Run precompilation for this target
        if ! run_precompile_for_target "${target}" "${docker_image}"; then
            log_error "Failed to precompile for ${target}"
            failed_targets+=("${target}")
            continue
        fi

        successful_targets+=("${target}")
    done

    echo
    log_info "=== Build Summary ==="

    if [ ${#successful_targets[@]} -gt 0 ]; then
        log_info "Successfully built targets (${#successful_targets[@]}):"
        for target in "${successful_targets[@]}"; do
            echo "  ✓ ${target}"
        done
    fi

    if [ ${#failed_targets[@]} -gt 0 ]; then
        log_warn "Failed targets (${#failed_targets[@]}):"
        for target in "${failed_targets[@]}"; do
            echo "  ✗ ${target}"
        done
    fi

    # List generated artifacts
    echo
    if ls "${CACHE_DIR}"/*.tar.gz &> /dev/null; then
        log_info "Generated artifacts in ${CACHE_DIR}:"
        ls -lh "${CACHE_DIR}"/*.tar.gz | awk '{print "  " $9 " (" $5 ")"}'
    else
        log_warn "No artifacts found in ${CACHE_DIR}"
    fi

    echo
    log_info "Usage:"
    log_info "  $0              - Build precompiled binaries for all targets"
    log_info "  $0 <target>     - Build specific target only"
    log_info ""
    log_info "To test (run locally after building artifacts):"
    log_info "  mix test        - Tests native build with FFTW linking"

    # Exit with error if any targets failed
    if [ ${#failed_targets[@]} -gt 0 ]; then
        exit 1
    fi
}

# Run main function
main "$@"
