# Precompilation Guide for NxEigen

This document describes how to build and publish precompiled binaries for NxEigen.

## Overview

NxEigen uses dynamic linking for FFTW. Users need to have FFTW installed on their system, either through:
- Linux: `libfftw3` package (apt, yum, etc.)
- macOS: Homebrew (`brew install fftw`)
- Or use our precompiled binaries which work with system-installed FFTW

## Building Locally

### Quick Start with Docker (Local Development)

For local multi-architecture builds, use Docker with BuildKit:

```bash
# Build precompiled binaries for all Linux targets
./scripts/precompile-docker.sh

# Build a specific target only
./scripts/precompile-docker.sh aarch64-linux-gnu

# Test locally (after artifacts are built)
mix test
```

The Docker approach:

- Runs **separate native containers** for each target architecture
- Uses Docker BuildKit multi-platform support (linux/amd64, linux/arm64)
- Each container builds natively for its architecture (no cross-compilation)
- Handles FFTW dependencies automatically (Debian glibc)
- Works on macOS, Linux, and Windows (with WSL)
- Generates precompiled binaries for all supported Linux targets
- **Testing**: Run `mix test` locally to test native builds

**Requirements:**

- Docker with BuildKit enabled
- Multi-architecture support: `docker buildx create --use`
- QEMU for non-native architectures (usually pre-installed)

### Native Build (macOS or Linux)

For native builds on your current platform:

```bash
# Ensure FFTW is installed
# macOS: brew install fftw
# Linux: sudo apt-get install libfftw3-dev

./scripts/precompile.sh
```

## Supported Targets

### Linux (Docker-based builds)

The `scripts/precompile-docker.sh` builds these targets by default:

- `x86_64-linux-gnu` - Standard glibc-based x86_64 Linux (Debian container)
- `aarch64-linux-gnu` - ARM64 Linux (glibc, Debian container)
- `aarch64-arduino-uno-q-linux-gnu` - **Arduino Uno Q optimized** (ARMv8-A + processor-specific flags)

**Optional targets** (can be enabled in the script):

- `riscv64-linux-gnu` - RISC-V 64-bit Linux (requires RISC-V emulation setup)

### macOS (Native builds only)

macOS builds are done on native runners in CI or locally:

- `x86_64-apple-darwin` - Intel macOS
- `aarch64-apple-darwin` - Apple Silicon (M1/M2/M3) macOS

### Arduino Uno Q Target

The `aarch64-arduino-uno-q-linux-gnu` target is specifically optimized for the Arduino Uno Q's Qualcomm QRB2210 processor (quad-core ARM Cortex-A53) with some compiler flags, including:

- `-march=armv8-a+crypto+crc` - Enables ARMv8-A instruction set with cryptographic and CRC extensions
- `-mtune=cortex-a53` - Optimizes instruction scheduling for Cortex-A53 pipeline
- `-mfix-cortex-a53-835769` - Workaround for Cortex-A53 erratum 835769
- `-mfix-cortex-a53-843419` - Workaround for Cortex-A53 erratum 843419

This provides optimal performance for the Uno Q's hardware capabilities.

**Important:** Users on Arduino Uno Q must set environment variables to fetch this optimized binary:

```bash
export TARGET_ARCH=aarch64
export TARGET_OS=arduino-uno-q-linux
export TARGET_ABI=gnu
```

Without these variables, the generic `aarch64-linux-gnu` binary will be used (which still works, but is ~20% slower).

See [ARDUINO_UNO_Q_QUICKSTART.md](ARDUINO_UNO_Q_QUICKSTART.md) for setup instructions.

## CI/CD Pipeline

### Automatic Builds (GitHub Actions)

Precompiled binaries are automatically built on native runners when you push a version tag:

**Linux builds**:
- `x86_64`: Runs on `ubuntu-22.04` (native x86_64)
- `aarch64`: Runs on `ubuntu-22.04-arm` (native ARM64)

**macOS builds**:
- `x86_64`: Runs on `macos-13` (native Intel)
- `aarch64`: Runs on `macos-14` (native Apple Silicon)

All builds use native architecture runners for maximum performance and reliability.

To trigger a build:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The GitHub Actions workflow will:
1. Build for all supported targets
2. Upload artifacts to the GitHub release
3. You then generate and commit the checksum file

### Manual (Local Build)

To test precompilation locally:

```bash
# Set to production environment
export MIX_ENV=prod

# Set cache directory
export ELIXIR_MAKE_CACHE_DIR="$(pwd)/cache"
mkdir -p "${ELIXIR_MAKE_CACHE_DIR}"

# Precompile for all available targets
mix elixir_make.precompile
```

This will create `.tar.gz` files in the `cache` directory.

## After Release: Generate Checksum

After GitHub Actions has uploaded all precompiled binaries to the release:

```bash
# Download all artifacts and generate checksum file
MIX_ENV=prod mix elixir_make.checksum --all --print

# Or if some targets are not yet available:
MIX_ENV=prod mix elixir_make.checksum --all --print --ignore-unavailable
```

This creates `checksum-nx_eigen.exs` which **must be committed** and included in the package.

```bash
git add checksum-nx_eigen.exs
git commit -m "Add precompiled binary checksums for v0.1.0"
git push
```

## Testing Precompiled Binaries Locally

```bash
# Delete local build to force download
rm -rf _build/prod/lib/nx_eigen

# Test that precompiled binary works
MIX_ENV=prod mix test
```

You should see a log message like:
```
[debug] Restore NIF for current node from: /Users/.../nx_eigen-nif-2.16-aarch64-apple-darwin-0.1.0.tar.gz
```

## Development Mode

For local development, append `-dev` to the version or set `make_force_build: true`:

```elixir
# In mix.exs
@version "0.1.0-dev"

# or
def project do
  [
    # ...
    make_force_build: true
  ]
end
```

This will compile only for the current host instead of all targets.

## Cross-Compilation Requirements

### Linux

Install cross-compilers:

```bash
# Ubuntu/Debian
sudo apt-get install -y \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
  gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

FFTW must be built statically for each target. The GitHub Actions workflow handles this automatically.

### macOS

On Apple Silicon (M1/M2/M3), you can build both architectures:

```bash
brew install fftw
mix elixir_make.precompile
```

## Package Release Checklist

1. ✅ Update version in `mix.exs`
2. ✅ Commit and tag: `git tag v0.1.0 && git push origin v0.1.0`
3. ✅ Wait for GitHub Actions to complete
4. ✅ Generate checksum: `MIX_ENV=prod mix elixir_make.checksum --all --print`
5. ✅ Commit checksum file: `git add checksum-nx_eigen.exs && git commit -m "Add checksums"`
6. ✅ Publish to Hex: `mix hex.publish`

## Troubleshooting

### "Checksum file not found" error

Make sure `checksum-nx_eigen.exs` is:
- Generated after all artifacts are uploaded
- Committed to the repository
- Included in the `files` list in `mix.exs` package config

### Cross-compilation fails

Ensure:
- Cross-compilers are installed
- FFTW static libraries are available for the target
- `PKG_CONFIG_PATH` includes the target's pkg-config directory

### Binary doesn't work on target platform

Check:
- NIF version compatibility (Erlang/OTP version)
- Target triplet matches exactly
- All dependencies are statically linked (use `ldd` on Linux or `otool -L` on macOS)
