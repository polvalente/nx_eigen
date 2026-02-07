#!/usr/bin/env bash
set -euo pipefail

# Docker-based precompilation script for nx_eigen
# This script builds a Docker image with cross-compilation toolchains
# and runs mix elixir_make.precompile inside it

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${ELIXIR_MAKE_CACHE_DIR:-${PROJECT_ROOT}/cache}"
DOCKER_IMAGE="nx_eigen_builder:latest"

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

# Build the Docker image with cross-compilation toolchains
build_docker_image() {
    log_step "Building Docker image with cross-compilation toolchains"
    
    cd "${PROJECT_ROOT}"
    
    if ! docker buildx build \
        --file Dockerfile.precompile \
        --tag "${DOCKER_IMAGE}" \
        --load \
        .; then
        log_error "Failed to build Docker image"
        return 1
    fi
    
    log_info "Docker image built successfully"
}

# Run precompilation inside Docker
run_precompile_in_docker() {
    log_step "Running precompilation inside Docker container"
    
    mkdir -p "${CACHE_DIR}"
    
    if ! docker run --rm \
        -v "${PROJECT_ROOT}:/work" \
        -w /work \
        -e MIX_ENV=prod \
        -e ELIXIR_MAKE_CACHE_DIR=/work/cache \
        "${DOCKER_IMAGE}" \
        bash -c "
          set -e
          echo 'Installing Hex and Rebar...'
          mix local.hex --force
          mix local.rebar --force
          
          echo 'Getting dependencies...'
          mix deps.get
          
          echo 'Running precompilation...'
          mix elixir_make.precompile
          
          echo 'Precompilation complete!'
        "; then
        log_error "Precompilation failed"
        return 1
    fi
    
    log_info "Precompilation complete"
}

# Run tests inside Docker
run_tests_in_docker() {
    log_step "Running tests inside Docker container"
    
    if ! docker run --rm \
        -v "${PROJECT_ROOT}:/work" \
        -w /work \
        -e MIX_ENV=test \
        "${DOCKER_IMAGE}" \
        bash -c "
          set -e
          mix local.hex --force
          mix local.rebar --force
          mix deps.get
          mix test
        "; then
        log_warn "Tests failed (this may be expected if you haven't built the native version)"
        return 1
    fi
    
    log_info "Tests passed"
}

# Main execution
main() {
    local run_tests="${1:-no}"
    
    log_info "Starting Docker-based precompilation"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Cache directory: ${CACHE_DIR}"
    log_info "Docker image: ${DOCKER_IMAGE}"
    
    # Check prerequisites
    check_docker || exit 1
    
    # Build Docker image
    build_docker_image || exit 1
    
    # Run precompilation
    run_precompile_in_docker || exit 1
    
    # Optionally run tests
    if [ "$run_tests" = "test" ] || [ "$run_tests" = "--test" ]; then
        run_tests_in_docker || true
    fi
    
    log_info "All builds complete!"
    
    # List generated artifacts
    if ls "${CACHE_DIR}"/*.tar.gz &> /dev/null; then
        log_info "Generated artifacts:"
        ls -lh "${CACHE_DIR}"/*.tar.gz | sed 's/^/  /'
    else
        log_warn "No artifacts found in ${CACHE_DIR}"
    fi
    
    echo
    log_info "To run tests: $0 test"
}

# Run main function
main "$@"
