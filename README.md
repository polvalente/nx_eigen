# NxEigen

An Elixir Nx backend that binds the [Eigen C++ library](https://eigen.tuxfamily.org) for efficient numerical computing on embedded systems, specifically targeting the Arduino Uno Q.

## Features

- **Complete Nx.Backend implementation** - All required callbacks implemented
- **Efficient linear algebra** - Uses Eigen's optimized matrix operations
- **FFT support** - Fast Fourier Transform via FFTW
- **All Nx types** - Support for u8-u64, s8-s64, f32/f64, c64/c128
- **Embedded-friendly** - Bitwise operations, integer math, and efficient memory usage
- **No template metaprogramming nonsense** - Clean, straightforward C++ implementations

## Dependencies

### Required

- **Eigen** (≥3.4.0) - C++ template library for linear algebra
- **FFTW3** - Fast Fourier Transform library
- **Elixir** (≥1.14)
- **Erlang/OTP** (≥25)

### Installation

#### Using System Packages

**macOS:**

```bash
brew install fftw
```

**Ubuntu/Debian:**

```bash
sudo apt-get install libfftw3-dev
```

**Arch Linux:**

```bash
sudo pacman -S fftw
```

#### Using Local Directories

You can specify local installations of Eigen and FFTW:

```bash
# Set environment variables before compiling
export EIGEN_DIR=/path/to/eigen
export FFTW_DIR=/path/to/fftw

mix deps.get
mix compile
```

#### Cross-compilation

This project builds a NIF (`priv/libnx_eigen.so`) via `make`. For cross-compilation you typically want to:

- **Set a toolchain**: `CROSSCOMPILE` (prefix) or `CXX` (full path)
- **Set the target OS** (so we don't add macOS-only linker flags): `TARGET_OS=Linux|Darwin`
- **FFTW note**: by default, when `CROSSCOMPILE` is set we build with `DISABLE_FFTW=1` (so `fft/ifft` are disabled and no FFTW headers/libs are needed). You can override with `DISABLE_FFTW=0 FFTW_DIR=...` if you have FFTW available for the target.
- **(If needed)** override `ERL_INCLUDE_DIR` to a matching Erlang/OTP include directory

Example (toolchain-prefix style):

```bash
export CROSSCOMPILE=aarch64-linux-gnu-
export TARGET_OS=Linux
export EIGEN_DIR=/path/to/eigen

mix deps.get
mix compile
```

If you're setting the compiler directly (and not setting `CROSSCOMPILE`), make sure to also disable FFTW explicitly:

```bash
DISABLE_FFTW=1 CXX=/path/to/aarch64-linux-gnu-g++ TARGET_OS=Linux SKIP_DOWNLOADS=1 mix compile
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
- The target must have **FFTW runtime** installed (or you must ship it alongside the `.so`)

On the **target** (Debian arm64), install deps:

```bash
sudo apt-get update
sudo apt-get install -y erlang-dev libfftw3-dev
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
export FFTW_DIR="$SYSROOT/usr"

make SKIP_DOWNLOADS=1 USE_CMAKE=1 \
  CMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu-sysroot.cmake \
  CMAKE_BUILD_DIR=$PWD/cmake-build-aarch64 \
  CMAKE_BUILD_TYPE=Release \
  CMAKE_ARGS="-DCMAKE_SYSROOT=$SYSROOT" \
  ERL_INCLUDE_DIR="$ERL_INCLUDE_DIR" \
  FFTW_DIR="$FFTW_DIR"
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
      "FFTW_DIR" => "/path/to/fftw",
      "CROSSCOMPILE" => "aarch64-linux-gnu-",
      "TARGET_OS" => "Linux"
    }
  ]
end
```

## Installation

Add `nx_eigen` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:nx, "~> 0.10"},
    {:nx_eigen, "~> 0.1.0"}
  ]
end
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

# FFT operations (requires FFTW)
signal = NxEigen.tensor([1.0, 2.0, 3.0, 4.0])
freq = Nx.fft(signal)
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

## Arduino Uno Q Support

The Arduino Uno Q features a Linux microprocessor alongside an STM32 microcontroller. This backend is designed to run on the Linux side, providing:

- Efficient numerical computing for sensor data processing
- Signal processing with FFT
- Matrix operations for control algorithms
- Bitwise operations for hardware interfacing

## License

Copyright (c) 2025

## Documentation

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc):

```bash
mix docs
```
