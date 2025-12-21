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

Or set them in your `mix.exs`:

```elixir
def project do
  [
    # ...
    make_env: %{
      "EIGEN_DIR" => "/path/to/eigen",
      "FFTW_DIR" => "/path/to/fftw"
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
