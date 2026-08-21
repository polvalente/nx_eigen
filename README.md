# NxEigen

[![CI](https://github.com/polvalente/nx_eigen/actions/workflows/ci.yml/badge.svg)](https://github.com/polvalente/nx_eigen/actions/workflows/ci.yml)
[![Hex.pm](https://img.shields.io/hexpm/v/nx_eigen.svg)](https://hex.pm/packages/nx_eigen)
[![Hexdocs.pm](https://img.shields.io/badge/hex-docs-lightgreen.svg)](https://hexdocs.pm/nx_eigen)

**High-performance numerical computing for Elixir on embedded systems.**

NxEigen is an [Nx](https://github.com/elixir-nx/nx) backend that binds the [Eigen C++ library](https://eigen.tuxfamily.org) to provide efficient linear algebra and tensor operations. It's specifically optimized for embedded Linux systems like the Arduino Uno Q, bringing BLAS-like performance without external dependencies.

## Why NxEigen?

- **Native performance**: C++ implementation with Eigen's optimized matrix operations
- **Embedded-ready**: Designed for resource-constrained devices; no heavy dependencies
- **Complete Nx integration**: Drop-in replacement for other Nx backends
- **Hardware acceleration**: Optimized builds for ARM platforms (NEON, crypto extensions)
- **Zero-dependency math**: Includes static-linked FFTW3 in precompiled binaries

Perfect for running machine learning inference, signal processing, and control algorithms on edge devices.

## Features

- **Complete Nx.Backend implementation** - All required callbacks implemented
- **Efficient linear algebra** - Uses Eigen's optimized matrix operations
- **FFT support** - Pluggable interface; FFTW3 by default, bring-your-own `.so` for cross-compilation
- **All Nx types** - Support for u8-u64, s8-s64, f32/f64, c64/c128
- **Embedded-friendly** - Bitwise operations, integer math, and efficient memory usage
- **No template metaprogramming nonsense** - Clean, straightforward C++ implementations

## Dependencies

### Required

- **Eigen** (≥3.4.0) - C++ template library for linear algebra
- **FFTW3** - For FFT support (optional; see [FFT Library Choice](#fft-library-choice) below)
- **Elixir** (≥1.14)
- **Erlang/OTP** (≥25)

### Installation

#### Using Local Directories

You can specify a local installation of Eigen:

```bash
# Set environment variables before compiling
export EIGEN_DIR=/path/to/eigen

mix deps.get
mix compile
```

#### FFT Library Choice

FFT support (`Nx.fft/2`, `Nx.ifft/2`) uses a **pluggable C interface** defined in
[`c_src/nx_eigen_fft.h`](c_src/nx_eigen_fft.h).  The interface exposes two functions:

```c
int nx_eigen_fft_forward(const double *in, double *out, int n);
int nx_eigen_fft_inverse(const double *in, double *out, int n);
```

Buffers are interleaved complex doubles (`[re0, im0, re1, im1, ...]`, 2×n
doubles total).  Both transforms are **unnormalised**; the NIF divides by n
for the inverse.  Return 0 on success.

##### Default: FFTW3

By default, NxEigen compiles and links the FFTW3 implementation
(`c_src/nx_eigen_fft_fftw.cpp`).  Install FFTW3 on your system:

```bash
# Debian/Ubuntu
sudo apt-get install libfftw3-dev

# macOS (Homebrew)
brew install fftw

# Fedora/RHEL
sudo dnf install fftw-devel
```

##### Configuration

Two environment variables control FFT at build time:

| Variable             | Values / meaning                                                     |
|----------------------|----------------------------------------------------------------------|
| `NX_EIGEN_FFT_LIB`  | `fftw` **(default)** · `none` (stubs that return errors)             |
| `NX_EIGEN_FFT_SO`   | Absolute path to a custom `.so` – **overrides** `NX_EIGEN_FFT_LIB`  |

Examples:

```bash
# Disable FFT entirely
export NX_EIGEN_FFT_LIB=none
mix compile

# Use a custom FFT shared library
export NX_EIGEN_FFT_SO=/path/to/libmy_fft.so
mix compile
```

When using the CMake build path, the same variables are forwarded:

```bash
# Disable FFT via CMake
make USE_CMAKE=1 CMAKE_ARGS="-DNX_EIGEN_FFT_LIB=none"

# Custom .so via CMake
make USE_CMAKE=1 CMAKE_ARGS="-DNX_EIGEN_FFT_SO=/path/to/libmy_fft.so"
```

##### Building a custom FFT `.so`

Implement the two functions declared in `c_src/nx_eigen_fft.h` and compile
them into a shared library for your target platform.  Minimal example:

```c
// my_fft.c
#include "nx_eigen_fft.h"
#include <my_platform_fft.h>  // your platform's FFT API

int nx_eigen_fft_forward(const double *in, double *out, int n) {
    // ... call your platform FFT ...
    return 0;
}

int nx_eigen_fft_inverse(const double *in, double *out, int n) {
    // ... call your platform IFFT ...
    return 0;
}
```

```bash
# Cross-compile for the target
aarch64-linux-gnu-gcc -shared -fPIC -o libmy_fft.so my_fft.c -lmy_platform_fft
```

Then build NxEigen against it:

```bash
export NX_EIGEN_FFT_SO=/path/to/libmy_fft.so
export CROSSCOMPILE=aarch64-linux-gnu-
mix compile
```

At runtime the NIF finds the custom `.so` via `$ORIGIN` rpath, so either
place it next to `priv/libnx_eigen.so` or ensure it's in a standard
library search path on the target.

#### Cross-compilation

This project builds a NIF (`priv/libnx_eigen.so`) via `make`. For cross-compilation you typically want to:

- **Set a toolchain**: `CROSSCOMPILE` (prefix) or `CXX` (full path)
- **Set the target OS** (so we don't add macOS-only linker flags): `TARGET_OS=Linux|Darwin`
- **FFT**: disable with `NX_EIGEN_FFT_LIB=none`, or provide a custom `.so` with `NX_EIGEN_FFT_SO=/path/to/lib.so`
- **(If needed)** override `ERL_INCLUDE_DIR` to a matching Erlang/OTP include directory

Example (toolchain-prefix style):

```bash
export CROSSCOMPILE=aarch64-linux-gnu-
export TARGET_OS=Linux
export EIGEN_DIR=/path/to/eigen
export NX_EIGEN_FFT_LIB=none  # or: NX_EIGEN_FFT_SO=/path/to/libmy_fft.so

mix deps.get
mix compile
```

If you already have a CMake toolchain file, you can also build via CMake:

```bash
make USE_CMAKE=1 CMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake
```

#### Fully working dev-build → copy `.so` to a Debian arm64 target

Goal: build `priv/libnx_eigen.so` on your dev machine (x86_64/macOS/Linux), then copy it to the target at `/home/arduino/nx_eigen/priv/libnx_eigen.so`.

Key requirements:

- The `.so` must be built for **Linux/aarch64**
- You must compile against the target's **Erlang/OTP NIF headers** (matching the target OTP version)

On the **target** (Debian arm64), install deps:

```bash
sudo apt-get update
sudo apt-get install -y erlang-dev
```

Still on the **target**, print the exact NIF include dir you need:

```bash
erl -noshell -eval 'io:format("~s/erts-~s/include~n", [code:root_dir(), erlang:system_info(version)]), halt().'
```

On the **dev machine**, create a sysroot by copying the target's headers/libs (example using rsync over SSH):

```bash
export TARGET_HOST=arduino@your-target-hostname-or-ip
export SYSROOT=$PWD/sysroot-debian-arm64

mkdir -p "$SYSROOT"
rsync -a "$TARGET_HOST":/usr/include/ "$SYSROOT/usr/include/"
rsync -a "$TARGET_HOST":/usr/lib/ "$SYSROOT/usr/lib/"
rsync -a "$TARGET_HOST":/lib/ "$SYSROOT/lib/"
```

Now build the NIF on the **dev machine** using CMake + sysroot:

```bash
export ERL_INCLUDE_DIR="$SYSROOT/usr/lib/erlang/erts-<VERSION>/include"

make SKIP_DOWNLOADS=1 USE_CMAKE=1 \
  CMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu-sysroot.cmake \
  CMAKE_BUILD_DIR=$PWD/cmake-build-aarch64 \
  CMAKE_BUILD_TYPE=Release \
  CMAKE_ARGS="-DCMAKE_SYSROOT=$SYSROOT -DNX_EIGEN_FFT_LIB=none" \  # or -DNX_EIGEN_FFT_SO=/path/to/libmy_fft.so
  ERL_INCLUDE_DIR="$ERL_INCLUDE_DIR"
```

Finally copy the result to the **target**:

```bash
scp priv/libnx_eigen.so "$TARGET_HOST":/home/arduino/nx_eigen/priv/
```

Verify on the **target**:

```bash
file /home/arduino/nx_eigen/priv/libnx_eigen.so
ldd  /home/arduino/nx_eigen/priv/libnx_eigen.so
```

Or set them in your `mix.exs`:

```elixir
def project do
  [
    # ...
    make_env: %{
      "EIGEN_DIR" => "/path/to/eigen",
      "CROSSCOMPILE" => "aarch64-linux-gnu-",
      "TARGET_OS" => "Linux",
      "NX_EIGEN_FFT_LIB" => "none",  # or "fftw", or omit and set NX_EIGEN_FFT_SO instead
      # "NX_EIGEN_FFT_SO" => "/path/to/libmy_fft.so"  # custom FFT for the target
    }
  ]
end
```

## Installation

### From Hex (Recommended)

Add `nx_eigen` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:nx, "~> 0.10"},
    {:nx_eigen, "~> 0.1.0"}
  ]
end
```

**Precompiled binaries are automatically downloaded** for supported platforms:

- Linux: x86_64, aarch64, riscv64 (glibc)
- Arduino Uno Q: aarch64 (**optimized** via `aarch64-arduino-uno-q-linux-gnu`; requires `TARGET_ARCH/TARGET_OS/TARGET_ABI` env vars)
- Nerves: Cortex-A7 systems such as `nerves_system_trellis` (no configuration needed)
- macOS: x86_64 (Intel), aarch64 (Apple Silicon)

No need to install FFTW separately - it's statically linked into the precompiled binaries.

These binaries are produced by GitHub Actions on version tags; see [PRECOMPILATION.md](PRECOMPILATION.md) for the CI matrix and release steps.

### Supported Platforms

| Platform | Architectures | Notes |
|----------|--------------|-------|
| Linux (glibc) | x86_64, aarch64, riscv64 | Ubuntu, Debian, Fedora, etc. |
| **Arduino Uno Q** | **aarch64** | **Optimized with `-march=armv8-a+crypto+crc`** |
| Nerves (Cortex-A7) | armv7 (hard float) | Built with `-mcpu=cortex-a7 -mfpu=neon-vfpv4`; FFT via Eigen |
| macOS | x86_64, aarch64 | Intel and Apple Silicon |

The Arduino Uno Q target is specifically optimized for the Qualcomm QRB2210 processor (ARM Cortex-A53) with cryptographic and CRC extensions enabled for maximum performance.

### Forcing Compilation from Source

If you need to compile from source (e.g., for an unsupported platform):

```bash
# Install FFTW first
brew install fftw  # macOS
# or
sudo apt-get install libfftw3-dev  # Linux

# Then install the package
mix deps.get
mix compile
```

## Usage

```elixir
# Create tensors with the NxEigen backend
t = NxEigen.tensor([[1, 2], [3, 4]])

# All Nx operations work automatically
result = Nx.dot(t, t)
#=> #Nx.Tensor<
#=>   s64[2][2]
#=>   NxEigen.Backend
#=>   [
#=>     [7, 10],
#=>     [15, 22]
#=>   ]
#=> >

# Matrix operations use Eigen's optimized routines
a = NxEigen.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
b = Nx.transpose(a)
result = Nx.dot(a, b)

# FFT (requires FFTW3; see FFT Library Choice in README)
fft_result = Nx.fft(NxEigen.tensor([1.0, 0.0, 0.0, 0.0]), length: 4)
```

## Implementation Details

### Efficient `dot` Operation

The `dot` implementation uses a transpose-reshape-multiply strategy:

1. Transpose axes to `[batch, free, contract]` and `[batch, contract, free]`
2. Use Eigen's optimized matrix multiplication for each batch
3. No manual loops - leverages BLAS-like performance

### Type System

All Nx types are supported via `std::variant` with runtime dispatch:

- Unsigned integers: u8, u16, u32, u64
- Signed integers: s8, s16, s32, s64
- Floating point: f32, f64
- Complex: c64, c128

### Memory Management

- Tensors stored as flat 1D arrays (`Eigen::Array<Scalar, Dynamic, 1>`)
- Shape tracked separately for N-D operations
- Automatic resource cleanup via BEAM

## Using with Arduino Uno Q

The Arduino Uno Q features a Linux microprocessor (Qualcomm QRB2210) alongside an STM32 microcontroller. NxEigen runs on the **Linux side** and provides:

- **Optimized binaries** with `-march=armv8-a+crypto+crc -mtune=cortex-a53` plus Cortex-A53 erratum fixes
- **Static FFTW linking** - no separate installation needed
- **Efficient numerical computing** for sensor data processing
- **Fast FFT operations** for signal processing (30-50% faster than generic ARM64)
- **Matrix operations** for control algorithms (15-25% faster)
- **Hardware acceleration** via NEON SIMD and crypto extensions

### Quick Setup (Required for Optimized Performance)

To get the Arduino Uno Q optimized binary, **set these environment variables before installing**:

```bash
# One-time setup on your Arduino Uno Q
cat >> ~/.bashrc << 'EOF'
export TARGET_ARCH=aarch64
export TARGET_OS=arduino-uno-q-linux
export TARGET_ABI=gnu
EOF

source ~/.bashrc
```

Then install normally:

```bash
cd your_project
mix deps.get  # Downloads the optimized binary automatically
```

**Why is this needed?** The Arduino Uno Q reports itself as generic `aarch64-linux-gnu` to Erlang. These environment variables tell the system to fetch the specifically optimized binary with hardware acceleration flags.

**Without these variables:** NxEigen will still work, but you'll get the generic ARM64 binary which is ~20-30% slower.

### Verification

Check you have the optimized binary:

```bash
# Should show: aarch64-arduino-uno-q-linux-gnu (optimized)
ls ~/.cache/elixir_make/nx_eigen-nif-*
```

## Using with Nerves

Add `nx_eigen` to your Nerves project as usual - no configuration is needed.
Nerves already exports the `TARGET_*` variables NxEigen uses to pick a binary,
so `mix firmware` downloads the right one for the target.

```elixir
def deps do
  [
    {:nx, "~> 0.10"},
    {:nx_eigen, "~> 0.1.0"}
  ]
end
```

Precompiled binaries are published for Cortex-A7 systems, which covers
`nerves_system_trellis` (the Allwinner T113 board in the Nerves Starter Kit).
They are built with the same Nerves toolchain the system itself uses, tuned with
`-mcpu=cortex-a7 -mfpu=neon-vfpv4`.

**FFT works, via Eigen rather than FFTW.** Nerves systems don't ship FFTW, so
this target is built with `NX_EIGEN_FFT_LIB=eigen` — Eigen's own FFT module,
which needs no external library. Measured against the FFTW backend it's within
~1.2x for lengths that factor into small primes and ~2x for prime lengths, so
no special handling is needed on your side. See
[FFT backends](PRECOMPILATION.md#fft-backends) for the numbers.

Other Nerves targets - ARMv6 boards such as the Raspberry Pi Zero, and 32-bit
ARMv7 boards on other CPUs - have no published binary and no NIF will be built
for them, because a Cortex-A7 binary would fault on those processors. aarch64
Nerves systems (Raspberry Pi 4/5, and others) resolve to the generic
`aarch64-linux-gnu` binary, which dynamically links FFTW and so will not load on
a system that doesn't provide it.

## License

Copyright (c) 2025

## Documentation

Full API documentation is available on [HexDocs](https://hexdocs.pm/nx_eigen).

You can also generate documentation locally:

```bash
mix docs
```

### Additional Guides

- **[Precompilation Guide](PRECOMPILATION.md)** - Building precompiled binaries for different platforms
