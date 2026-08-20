# Precompilation Guide for NxEigen

This document describes how to build and publish precompiled binaries for NxEigen.

## Overview

NxEigen uses dynamic linking for FFTW. Users need to have FFTW installed on their system, either through:
- Linux: `libfftw3` package (apt, yum, etc.)
- macOS: Homebrew (`brew install fftw`)
- Or use our precompiled binaries which work with system-installed FFTW

## FFT backends

`NX_EIGEN_FFT_LIB` selects the implementation compiled in behind
`c_src/nx_eigen_fft.h`:

| Value   | Implementation | External dependency |
| ------- | -------------- | ------------------- |
| `fftw`  | FFTW3 (default) | `libfftw3`, `libfftw3f` at runtime |
| `eigen` | Eigen's FFT module (vendored kissfft, MPL-2.0) | none |
| `none`  | Stubs that raise | none |

`NX_EIGEN_FFT_SO` overrides all of them with a path to your own shared library
exporting the same symbols.

The `eigen` backend exists for targets with no FFTW, and is what the Nerves
target uses. Measured back to back on x86_64, f64, per call:

| Length        | FFTW     | Eigen    |
| ------------- | -------- | -------- |
| 64            | 7 µs     | 2 µs     |
| 1024          | 21 µs    | 20 µs    |
| 4096          | 0.119 ms | 0.127 ms |
| 65536         | 2.39 ms  | 2.78 ms  |
| 1021 (prime)  | 0.099 ms | 0.201 ms |
| 8191 (prime)  | 1.16 ms  | 2.10 ms  |
| 65521 (prime) | 9.19 ms  | 21.1 ms  |

Within ~1.2x for lengths kissfft factors directly, and ~2x at worst for prime
lengths. `eigen` wins outright at small sizes because the FFTW backend builds
and destroys a plan on every call (`FFTW_ESTIMATE`), while the Eigen one caches
plans per scheduler thread.

kissfft only has butterflies for radix 2/3/4/5 and its generic fallback costs
O(n · p) for largest prime factor p — an unusable O(n²) at prime lengths, which
measured 9.6 *seconds* for n=65521 before Bluestein's algorithm was added for
those lengths. `c_src/nx_eigen_fft_eigen.cpp` switches to Bluestein once the
largest prime factor exceeds 64, which is roughly where the two costs cross.

Note that the FFT NIFs are registered without dirty-scheduler flags
(`FINE_NIF(fft_nif, 0)`), so a large transform occupies a normal scheduler for
its duration on any backend.

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

## Supported Targets

### Linux (Docker-based builds)

The `scripts/precompile-docker.sh` builds these targets by default:

- `x86_64-linux-gnu` - Standard glibc-based x86_64 Linux (Debian container)
- `aarch64-linux-gnu` - ARM64 Linux (glibc, Debian container)
- `aarch64-arduino-uno-q-linux-gnu` - **Arduino Uno Q optimized** (ARMv8-A + processor-specific flags)

**Optional targets** (can be enabled in the script):

- `riscv64-linux-gnu` - RISC-V 64-bit Linux (requires RISC-V emulation setup)

### Nerves (cross-compiled)

- `armv7-cortex-a7-linux-gnueabihf` - Nerves systems on a Cortex-A7, such as
  `nerves_system_trellis` (Allwinner T113, the Nerves Starter Kit board)

This target is cross-compiled with the same Nerves toolchain the system is built
with (`armv7_nerves_linux_gnueabihf`), so the glibc and libstdc++ it links
against match those in the system.

Nerves systems do not ship FFTW, so `mix.exs` builds this target with
`NX_EIGEN_FFT_LIB=eigen` — Eigen's own FFT module, which needs no external
library. See [FFT backends](#fft-backends) for the trade-off.

To build it locally, put the toolchain on your `PATH` and run:

```bash
curl -fsSL https://github.com/nerves-project/toolchains/releases/download/v15.3.0/nerves_toolchain_armv7_nerves_linux_gnueabihf-linux_x86_64-15.3.0-9917D70.tar.xz | tar -xJ
export PATH="$(pwd)/nerves_toolchain_armv7_nerves_linux_gnueabihf/bin:${PATH}"

MIX_ENV=prod \
  PRECOMPILE_TARGET=armv7-cortex-a7-linux-gnueabihf \
  ELIXIR_MAKE_CACHE_DIR="$(pwd)/cache" \
  mix elixir_make.precompile
```

The `scripts/precompile-docker.sh` flow does not cover this target: it builds
natively in a container per architecture, while this one is cross-compiled.

#### How Nerves devices resolve a target

Nerves exports `TARGET_ARCH=arm`, `TARGET_OS=linux` and `TARGET_ABI=gnueabihf`
for *every* 32-bit ARM board, so `cc_precompiler` resolves ARMv6 and ARMv7
devices to the same `arm-linux-gnueabihf` triplet. `NxEigen.Precompiler` (in
`precompiler.exs`) refines that triplet using `TARGET_CPU`:

| `TARGET_CPU`   | Target                            | Published |
| -------------- | --------------------------------- | --------- |
| `cortex_a7`    | `armv7-cortex-a7-linux-gnueabihf` | Yes       |
| `arm1176*`     | `armv6-linux-gnueabihf`           | No        |
| anything else  | `arm-linux-gnueabihf`             | No        |

Unpublished targets fall back to `:ignore`, so boards we have not built for get
no NIF rather than a binary that faults with an illegal instruction.

When adding a target, add it to **both** the compiler map in `mix.exs` and
`@published_targets` in `precompiler.exs` - the latter is what a consumer uses
to work out the download URL.

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

See the [Arduino Uno Q section in the README](README.md#using-with-arduino-uno-q) for detailed setup instructions.

## CI/CD Pipeline

### Automatic Builds (GitHub Actions)

Precompiled binaries are automatically built on native runners when you push a version tag:

**Linux builds**:
- `x86_64`: Runs on `ubuntu-22.04` (native x86_64)
- `aarch64`: Runs on `ubuntu-22.04-arm` (native ARM64)
- `aarch64-arduino-uno-q-linux-gnu`: Runs on `ubuntu-22.04-arm` (native ARM64, with `PRECOMPILE_TARGET` set)

**macOS builds**:

- `x86_64`: Runs on `macos-15-intel` (native Intel)
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
# Download all artifacts and generate checksum file.
# `NxEigen.Precompiler` reports every published target regardless of the host,
# so no PRECOMPILE_TARGET is needed here.
MIX_ENV=prod mix elixir_make.checksum --all --print

# Or if some targets are not yet available:
MIX_ENV=prod mix elixir_make.checksum --all --print --ignore-unavailable
```

This creates `checksum.exs` which **must be committed** and included in the package.

```bash
git add checksum.exs
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
[debug] Restore NIF for current node from: /Users/.../nx_eigen-nif-2.17-aarch64-apple-darwin-0.1.0.tar.gz
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
4. ✅ Generate checksum: `MIX_ENV=prod mix elixir_make.checksum --all --print` (wait for all CI jobs to finish first)
5. ✅ Commit checksum file: `git add checksum.exs && git commit -m "Add checksums for vX.Y.Z"`
6. ✅ Publish to Hex: `mix hex.publish`

## Troubleshooting

### "Checksum file not found" error

Make sure `checksum.exs` is:
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
