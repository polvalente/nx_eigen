#!/usr/bin/env bash
set -euo pipefail

# Test precompiled binaries for nx_eigen
# This script loads each precompiled binary and runs the test suite
# to verify they work correctly on their target platforms

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${ELIXIR_MAKE_CACHE_DIR:-${PROJECT_ROOT}/cache}"

# Define all supported targets and their Docker platforms
declare -A TARGETS=(
    ["x86_64-linux-gnu"]="linux/amd64"
    ["aarch64-linux-gnu"]="linux/arm64"
    ["aarch64-arduino-uno-q-linux-gnu"]="linux/arm64"
    ["x86_64-linux-musl"]="linux/amd64"
    ["aarch64-linux-musl"]="linux/arm64"
)

# Base images for different target types
declare -A BASE_IMAGES=(
    ["x86_64-linux-gnu"]="hexpm/elixir:1.18.1-erlang-27.3-debian-bullseye-20260112-slim"
    ["aarch64-linux-gnu"]="hexpm/elixir:1.18.1-erlang-27.3-debian-bullseye-20260112-slim"
    ["aarch64-arduino-uno-q-linux-gnu"]="hexpm/elixir:1.18.1-erlang-27.3-debian-bullseye-20260112-slim"
    ["x86_64-linux-musl"]="hexpm/elixir:1.18.1-erlang-27.3-alpine-3.21.0"
    ["aarch64-linux-musl"]="hexpm/elixir:1.18.1-erlang-27.3-alpine-3.21.0"
)

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
        return 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        return 1
    fi

    log_info "Docker is available"
    return 0
}

# Find precompiled binary for target
find_precompiled_binary() {
    local target="$1"

    # Look for the precompiled tarball in cache directory
    # Format: nx_eigen-nif-2.17-${target}-*.tar.gz
    local pattern="${CACHE_DIR}/nx_eigen-nif-*-${target}-*.tar.gz"
    local found=$(ls ${pattern} 2>/dev/null | head -n 1)

    if [[ -n "$found" ]]; then
        echo "$found"
        return 0
    fi

    log_error "No precompiled binary found for ${target}"
    log_error "Expected pattern: ${pattern}"
    return 1
}

# Build test Docker image for a specific platform
build_test_image() {
    local platform="$1"
    local target="$2"
    local base_image="${BASE_IMAGES[$target]}"
    local tag="nx_eigen_test:${target}"

    log_step "Building test image for ${target} on ${platform}"

    cd "${PROJECT_ROOT}"

    if ! docker buildx build \
        --file Dockerfile.test \
        --platform "${platform}" \
        --tag "${tag}" \
        --build-arg BASE_IMAGE="${base_image}" \
        --build-arg TARGETPLATFORM="${platform}" \
        --build-arg PRECOMPILE_TARGET="${target}" \
        --load \
        .; then
        log_error "Failed to build test image for ${target}"
        return 1
    fi

    log_info "Test image built successfully for ${target}"
}

# Test a precompiled binary
test_precompiled_binary() {
    local target="$1"
    local binary_path="$2"
    local platform="$3"
    local docker_image="nx_eigen_test:${target}"

    log_step "Testing precompiled binary for ${target}"
    log_info "Binary: ${binary_path}"

    # Extract just the filename for copying
    local binary_filename=$(basename "${binary_path}")

    # Create a temporary directory for test context
    local temp_dir=$(mktemp -d)
    trap "rm -rf ${temp_dir}" RETURN

    # Extract the tarball to get the .so file
    log_info "Extracting precompiled binary..."
    tar -xzf "${binary_path}" -C "${temp_dir}"

    # Find the .so file (should be in the extracted directory)
    local so_file=$(find "${temp_dir}" -name "libnx_eigen.so" | head -n 1)

    if [[ -z "$so_file" ]]; then
        log_error "Could not find libnx_eigen.so in extracted tarball"
        return 1
    fi

    log_info "Found binary: ${so_file}"

    # Check what libraries the binary requires (if ldd is available)
    if command -v ldd &> /dev/null; then
        log_info "Binary dependencies:"
        ldd "${so_file}" 2>&1 | head -20 || true
    fi

    # Determine library path based on architecture
    local lib_path="/usr/lib"
    if [[ "$platform" == "linux/arm64" ]]; then
        lib_path="/usr/lib/aarch64-linux-gnu:/usr/lib"
    elif [[ "$platform" == "linux/amd64" ]]; then
        lib_path="/usr/lib/x86_64-linux-gnu:/usr/lib"
    fi

    # Run tests with the precompiled binary mounted directly to priv/
    if ! docker run --rm \
        -v "${so_file}:/work/priv/libnx_eigen.so:ro" \
        -e MIX_ENV=test \
        -e ELIXIR_MAKE_SKIP_COMPILATION_STEP=true \
        -e LD_LIBRARY_PATH="${lib_path}:/work/priv" \
        "${docker_image}"; then
        log_error "Tests failed for ${target}"
        return 1
    fi

    log_info "Tests passed for ${target}"
}

# Main execution
main() {
    local specific_target="${1:-}"

    log_info "Starting precompiled binary testing"
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

    # Check if cache directory exists and has binaries
    if [[ ! -d "${CACHE_DIR}" ]]; then
        log_error "Cache directory does not exist: ${CACHE_DIR}"
        log_error "Please run precompilation first: make precompile-docker"
        exit 1
    fi

    local binary_count=$(ls "${CACHE_DIR}"/*.tar.gz 2>/dev/null | wc -l)
    if [[ $binary_count -eq 0 ]]; then
        log_error "No precompiled binaries found in ${CACHE_DIR}"
        log_error "Please run precompilation first: make precompile-docker"
        exit 1
    fi

    log_info "Found ${binary_count} precompiled binaries"

    local failed_targets=()
    local successful_targets=()
    local skipped_targets=()

    # Test each target
    for target in "${!TARGETS[@]}"; do
        # If specific target requested, skip others
        if [[ -n "$specific_target" && "$target" != "$specific_target" ]]; then
            continue
        fi

        platform="${TARGETS[$target]}"

        log_step "Processing target: ${target} (${platform})"

        # Find precompiled binary
        if ! binary_path=$(find_precompiled_binary "${target}"); then
            log_warn "Skipping ${target} - no precompiled binary found"
            skipped_targets+=("${target}")
            continue
        fi

        # Build test Docker image
        if ! build_test_image "${platform}" "${target}"; then
            log_error "Failed to build test image for ${target}"
            failed_targets+=("${target}")
            continue
        fi

        # Run tests with precompiled binary
        if ! test_precompiled_binary "${target}" "${binary_path}" "${platform}"; then
            log_error "Tests failed for ${target}"
            failed_targets+=("${target}")
            continue
        fi

        successful_targets+=("${target}")
    done

    echo
    log_info "=== Test Summary ==="

    if [ ${#successful_targets[@]} -gt 0 ]; then
        log_info "Successfully tested targets (${#successful_targets[@]}):"
        for target in "${successful_targets[@]}"; do
            echo "  ✓ ${target}"
        done
    fi

    if [ ${#skipped_targets[@]} -gt 0 ]; then
        log_warn "Skipped targets (${#skipped_targets[@]}):"
        for target in "${skipped_targets[@]}"; do
            echo "  ⊘ ${target}"
        done
    fi

    if [ ${#failed_targets[@]} -gt 0 ]; then
        log_error "Failed targets (${#failed_targets[@]}):"
        for target in "${failed_targets[@]}"; do
            echo "  ✗ ${target}"
        done
    fi

    echo
    log_info "Usage:"
    log_info "  $0              - Test all precompiled binaries"
    log_info "  $0 <target>     - Test specific target only"
    log_info ""
    log_info "Available targets: ${!TARGETS[@]}"

    # Exit with error if any targets failed
    if [ ${#failed_targets[@]} -gt 0 ]; then
        exit 1
    fi

    # Also warn if all were skipped
    if [ ${#successful_targets[@]} -eq 0 ]; then
        log_warn "No targets were successfully tested"
        exit 1
    fi
}

# Run main function
main "$@"
