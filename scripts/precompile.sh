#!/usr/bin/env bash
set -e

# Precompile script for nx_eigen
# This script downloads Eigen and creates precompiled NIF binaries.
# FFTW is dynamically linked and expected to be available on the system.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${ELIXIR_MAKE_CACHE_DIR:-${PROJECT_ROOT}/cache}"

# Versions
EIGEN_VERSION="3.4.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Create necessary directories
mkdir -p "${CACHE_DIR}"

# Download and extract Eigen if not present
download_eigen() {
    local eigen_dir="${PROJECT_ROOT}/eigen-${EIGEN_VERSION}"

    if [ -d "${eigen_dir}" ]; then
        log_info "Eigen ${EIGEN_VERSION} already exists at ${eigen_dir}"
        return 0
    fi

    log_info "Downloading Eigen ${EIGEN_VERSION}..."
    cd "${PROJECT_ROOT}"
    curl -L -k "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" | tar xz
    log_info "Eigen ${EIGEN_VERSION} downloaded and extracted"
}

# Check if FFTW is installed
check_fftw() {
    log_info "Checking for FFTW installation..."
    
    if pkg-config --exists fftw3 fftw3f 2>/dev/null; then
        local version=$(pkg-config --modversion fftw3)
        log_info "Found FFTW ${version} via pkg-config"
        return 0
    fi
    
    # Try to find libraries directly
    case "$(uname -s)" in
        Darwin)
            if [ -f "/opt/homebrew/lib/libfftw3.dylib" ] || [ -f "/usr/local/lib/libfftw3.dylib" ]; then
                log_info "Found FFTW in Homebrew installation"
                return 0
            fi
            ;;
        Linux)
            if ldconfig -p 2>/dev/null | grep -q libfftw3; then
                log_info "Found FFTW in system libraries"
                return 0
            fi
            ;;
    esac
    
    log_error "FFTW not found. Please install it:"
    case "$(uname -s)" in
        Darwin)
            log_error "  brew install fftw"
            ;;
        Linux)
            log_error "  sudo apt-get install libfftw3-dev  (Debian/Ubuntu)"
            log_error "  sudo yum install fftw-devel        (RHEL/CentOS)"
            ;;
    esac
    return 1
}

# Main precompilation flow
main() {
    log_info "Starting precompilation process..."
    log_info "Cache directory: ${CACHE_DIR}"
    
    # Check dependencies
    check_fftw || exit 1
    download_eigen
    
    # Run Mix precompiler
    log_info "Running Mix precompiler..."
    cd "${PROJECT_ROOT}"
    
    MIX_ENV=prod \
    ELIXIR_MAKE_CACHE_DIR="${CACHE_DIR}" \
    mix elixir_make.precompile
    
    log_info "Precompilation complete!"
    log_info "Artifacts in: ${CACHE_DIR}"
    
    # List generated artifacts
    if ls "${CACHE_DIR}"/*.tar.gz 1> /dev/null 2>&1; then
        log_info "Generated artifacts:"
        ls -lh "${CACHE_DIR}"/*.tar.gz
    fi
}

# Run main function
main "$@"
