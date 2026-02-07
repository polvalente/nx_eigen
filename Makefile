ERL_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]), halt().')

# Cross-compilation configuration
# - Set CROSSCOMPILE to a toolchain prefix (e.g. aarch64-linux-gnu-)
# - Or set CXX directly (e.g. CXX=/path/to/clang++)
CROSSCOMPILE ?=
# Note: GNU Make has a built-in default for CXX; `?=` won't override it.
# Only derive CXX from CROSSCOMPILE when the user didn't set CXX explicitly.
ifeq ($(origin CXX),default)
  CXX = $(CROSSCOMPILE)g++
endif

# Eigen configuration
# Set EIGEN_DIR to use a local installation, otherwise download automatically
EIGEN_VERSION = 3.4.0
EIGEN_DIR ?= $(CURDIR)/eigen-$(EIGEN_VERSION)
EIGEN_INCLUDE = $(EIGEN_DIR)
FINE_INCLUDE = $(CURDIR)/deps/fine/c_include

# FFT library choice
# The NIF calls a pluggable C interface (see c_src/nx_eigen_fft.h).
#
# NX_EIGEN_FFT_LIB selects which implementation is compiled in:
#   fftw  - Use FFTW3 (default for native builds)
#   none  - Disable FFT support (stubs that return errors)
#
# NX_EIGEN_FFT_SO overrides NX_EIGEN_FFT_LIB entirely: set it to the
# path of a shared library that exports nx_eigen_fft_forward/inverse
# (see c_src/nx_eigen_fft.h for the contract).  Useful when cross-
# compiling with a custom FFT backend.
NX_EIGEN_FFT_LIB ?= fftw
NX_EIGEN_FFT_SO  ?=

CFLAGS = -fPIC -I$(ERL_INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(FINE_INCLUDE) -Ic_src -O3 -std=c++17
LDFLAGS = -shared

# Resolve FFT sources and link flags
FFT_SRCS =
FFT_LDFLAGS =
FFT_CFLAGS =

ifneq ($(NX_EIGEN_FFT_SO),)
  # Custom shared library – link directly against it
  FFT_LDFLAGS = $(NX_EIGEN_FFT_SO) -Wl,-rpath,'$$ORIGIN'
else ifeq ($(NX_EIGEN_FFT_LIB),fftw)
  FFT_SRCS = c_src/nx_eigen_fft_fftw.cpp

  # Determine pkg-config command (handle cross-compilation)
  PKG_CONFIG ?= pkg-config

  # Try pkg-config first (handles Homebrew installations properly)
  PKG_CONFIG_FFTW3 := $(shell $(PKG_CONFIG) --exists fftw3 2>/dev/null && echo yes)
  ifeq ($(PKG_CONFIG_FFTW3),yes)
    FFT_CFLAGS = $(shell $(PKG_CONFIG) --cflags fftw3 fftw3f)
    # For static linking, use --libs-only-L and link .a files directly
    FFTW_LIB_DIR := $(shell $(PKG_CONFIG) --variable=libdir fftw3)
    FFT_LDFLAGS = $(FFTW_LIB_DIR)/libfftw3.a $(FFTW_LIB_DIR)/libfftw3f.a
  else
    # Fallback: try to find static libraries in standard locations
    # This is useful for cross-compilation where pkg-config might not be configured
    ifneq ($(CROSSCOMPILE),)
      # Check for TARGET environment variable for special targets like Arduino Uno Q
      ifdef TARGET
        SYSROOT ?= /usr/$(TARGET)
      else
        # Cross-compilation mode - look for static libs in sysroot
        SYSROOT ?= /usr/$(CROSSCOMPILE:%-=%)
      endif
      ifneq ($(wildcard $(SYSROOT)/lib/libfftw3.a),)
        FFT_CFLAGS = -I$(SYSROOT)/include
        FFT_LDFLAGS = $(SYSROOT)/lib/libfftw3.a $(SYSROOT)/lib/libfftw3f.a
      else
        FFT_LDFLAGS = -lfftw3 -lfftw3f
      endif
    else
      FFT_LDFLAGS = -lfftw3 -lfftw3f
    endif
  endif
else ifeq ($(NX_EIGEN_FFT_LIB),none)
  FFT_SRCS = c_src/nx_eigen_fft_none.cpp
else
  $(error Unsupported NX_EIGEN_FFT_LIB value: $(NX_EIGEN_FFT_LIB). Use "fftw", "none", or set NX_EIGEN_FFT_SO.)
endif

UNAME_S := $(shell uname -s)
TARGET_OS ?= $(UNAME_S)
ifeq ($(TARGET_OS),Darwin)
	LDFLAGS += -undefined dynamic_lookup
endif

LIB_NAME = priv/libnx_eigen.so

# Optional CMake build (useful for cross-compilation via toolchain files)
USE_CMAKE ?= 0
CMAKE ?= cmake
CMAKE_BUILD_DIR ?= $(CURDIR)/cmake-build
CMAKE_BUILD_TYPE ?= Release
CMAKE_TOOLCHAIN_FILE ?=
CMAKE_ARGS ?=
SKIP_DOWNLOADS ?= 0

all: check-deps priv $(LIB_NAME)

# Check dependencies without rebuilding
check-deps:
	@if [ "$(SKIP_DOWNLOADS)" != "1" ] && [ ! -d "$(EIGEN_DIR)" ]; then \
		echo "Downloading Eigen $(EIGEN_VERSION)..."; \
		curl -L -k https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/eigen-$(EIGEN_VERSION).tar.gz | tar xz || \
		(echo "Failed to download Eigen. Please install manually or set EIGEN_DIR=/path/to/eigen"; exit 1); \
	fi

priv:
	@mkdir -p priv

$(LIB_NAME): c_src/nx_eigen_nif.cpp c_src/nx_eigen_fft.h $(FFT_SRCS) | check-deps priv
ifeq ($(USE_CMAKE),1)
	$(CMAKE) -S $(CURDIR) -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		$(if $(CMAKE_TOOLCHAIN_FILE),-DCMAKE_TOOLCHAIN_FILE=$(CMAKE_TOOLCHAIN_FILE),) \
		$(CMAKE_ARGS) \
		-DERL_INCLUDE_DIR=$(ERL_INCLUDE_DIR) -DEIGEN_DIR=$(EIGEN_DIR) -DFINE_INCLUDE=$(FINE_INCLUDE) \
		-DNX_EIGEN_FFT_LIB=$(NX_EIGEN_FFT_LIB) \
		$(if $(NX_EIGEN_FFT_SO),-DNX_EIGEN_FFT_SO=$(NX_EIGEN_FFT_SO),)
	$(CMAKE) --build $(CMAKE_BUILD_DIR) --config $(CMAKE_BUILD_TYPE)
else
	$(CXX) $(CFLAGS) $(FFT_CFLAGS) $(LDFLAGS) $(FFT_LDFLAGS) c_src/nx_eigen_nif.cpp $(FFT_SRCS) -o $(LIB_NAME)
endif

clean:
	rm -rf priv $(LIB_NAME) $(CMAKE_BUILD_DIR) eigen-$(EIGEN_VERSION)*

.PHONY: all clean check-deps

