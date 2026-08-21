# ERTS_INCLUDE_DIR is exported by cross-compilation environments such as Nerves
# and points at the target's erts includes rather than the build host's.
ifdef ERTS_INCLUDE_DIR
  ERL_INCLUDE_DIR ?= $(ERTS_INCLUDE_DIR)
else
  ERL_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]), halt().')
endif

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
FINE_INCLUDE ?= $(error FINE_INCLUDE is not set. Use mix compile instead of bare make.)

# FFT library choice
# The NIF calls a pluggable C interface (see c_src/nx_eigen_fft.h).
#
# NX_EIGEN_FFT_LIB selects which implementation is compiled in:
#   fftw  - Use FFTW3 (default for native builds)
#   eigen - Use Eigen's own FFT module; no external library, slower than FFTW
#   none  - Disable FFT support (stubs that return errors)
#
# NX_EIGEN_FFT_SO overrides NX_EIGEN_FFT_LIB entirely: set it to the
# path of a shared library that exports nx_eigen_fft_forward/inverse
# (see c_src/nx_eigen_fft.h for the contract).  Useful when cross-
# compiling with a custom FFT backend.
NX_EIGEN_FFT_LIB ?= fftw
NX_EIGEN_FFT_SO  ?=

# Inherit CXXFLAGS/LDFLAGS from the environment: cross-compilation environments
# such as Nerves use them to pass --sysroot and processor-specific flags, and a
# plain `=` assignment would discard them.
CFLAGS := $(CXXFLAGS) -fPIC -I$(ERL_INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(FINE_INCLUDE) -Ic_src -O3 -std=c++17
LDFLAGS := $(LDFLAGS) -shared -fvisibility=hidden

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

  # Try pkg-config first
  PKG_CONFIG_FFTW3 := $(shell $(PKG_CONFIG) --exists fftw3 2>/dev/null && echo yes)
  ifeq ($(PKG_CONFIG_FFTW3),yes)
    FFT_CFLAGS = $(shell $(PKG_CONFIG) --cflags fftw3 fftw3f)
    FFT_LDFLAGS = $(shell $(PKG_CONFIG) --libs fftw3 fftw3f)
    # Add rpath for runtime library discovery
    FFTW_LIBDIR := $(shell $(PKG_CONFIG) --variable=libdir fftw3 2>/dev/null)
    ifneq ($(FFTW_LIBDIR),)
      FFT_LDFLAGS += -Wl,-rpath,$(FFTW_LIBDIR)
    endif
  else
    # Fallback for cross-compilation with sysroot
    ifneq ($(CROSSCOMPILE),)
      # Nerves exports its sysroot; otherwise guess the /usr/<triple> layout
      # Debian's cross packages use, which only means anything when CROSSCOMPILE
      # is a bare prefix. Under Nerves it is an absolute path, and prefixing it
      # with /usr/ yielded paths like -I/usr//home/user/.nerves/...
      ifneq ($(NERVES_SDK_SYSROOT),)
        SYSROOT ?= $(NERVES_SDK_SYSROOT)
      else ifdef TARGET
        SYSROOT ?= /usr/$(TARGET)
      else ifeq ($(filter /%,$(CROSSCOMPILE)),)
        SYSROOT ?= /usr/$(CROSSCOMPILE:%-=%)
      else
        $(error Cannot locate a sysroot for FFTW: set SYSROOT, or NX_EIGEN_FFT_LIB=eigen to build without FFTW.)
      endif
      FFT_CFLAGS = -I$(SYSROOT)/usr/include -I$(SYSROOT)/include
      FFT_LDFLAGS = -L$(SYSROOT)/usr/lib -L$(SYSROOT)/lib -lfftw3 -lfftw3f -Wl,-rpath,$(SYSROOT)/usr/lib
    else
      # Default: assume system libraries with rpath for common locations
      FFT_LDFLAGS = -lfftw3 -lfftw3f -Wl,-rpath,/usr/lib -Wl,-rpath,/usr/local/lib
    endif
  endif
else ifeq ($(NX_EIGEN_FFT_LIB),eigen)
  FFT_SRCS = c_src/nx_eigen_fft_eigen.cpp
else ifeq ($(NX_EIGEN_FFT_LIB),none)
  FFT_SRCS = c_src/nx_eigen_fft_none.cpp
else
  $(error Unsupported NX_EIGEN_FFT_LIB value: $(NX_EIGEN_FFT_LIB). Use "fftw", "eigen", "none", or set NX_EIGEN_FFT_SO.)
endif

UNAME_S := $(shell uname -s)
TARGET_OS ?= $(UNAME_S)
ifeq ($(TARGET_OS),Darwin)
	LDFLAGS += -undefined dynamic_lookup
else
	# Add common library paths for runtime linking on Linux
	LDFLAGS += -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,'$$ORIGIN/../lib'
endif

# When invoked by elixir_make, MIX_APP_PATH points to _build/env/lib/app.
# Writing the .so there (rather than project-root priv/) is required so that
# cc_precompiler can find the artifact when building the release tarball.
PRIV_DIR = $(if $(MIX_APP_PATH),$(MIX_APP_PATH)/priv,priv)
LIB_NAME = $(PRIV_DIR)/libnx_eigen.so

# Optional CMake build (useful for cross-compilation via toolchain files)
USE_CMAKE ?= 0
CMAKE ?= cmake
CMAKE_BUILD_DIR ?= $(CURDIR)/cmake-build
CMAKE_BUILD_TYPE ?= Release
CMAKE_TOOLCHAIN_FILE ?=
CMAKE_ARGS ?=
SKIP_DOWNLOADS ?= 0

all: check-deps $(PRIV_DIR) $(LIB_NAME)

# Check dependencies without rebuilding
check-deps:
	@if [ "$(SKIP_DOWNLOADS)" != "1" ] && [ ! -d "$(EIGEN_DIR)" ]; then \
		echo "Downloading Eigen $(EIGEN_VERSION)..."; \
		curl -L -k https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/eigen-$(EIGEN_VERSION).tar.gz | tar xz || \
		(echo "Failed to download Eigen. Please install manually or set EIGEN_DIR=/path/to/eigen"; exit 1); \
	fi

# Only the directory being built into: creating a project-root priv/ when
# MIX_APP_PATH is set makes Mix point every _build/<target>_<env> priv at that
# one directory, and a host build then overwrites the target's .so.
$(PRIV_DIR):
	@mkdir -p "$(PRIV_DIR)"

$(LIB_NAME): c_src/nx_eigen_nif.cpp c_src/nx_eigen_fft.h $(FFT_SRCS) | check-deps $(PRIV_DIR)
ifeq ($(USE_CMAKE),1)
	$(CMAKE) -S $(CURDIR) -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		$(if $(CMAKE_TOOLCHAIN_FILE),-DCMAKE_TOOLCHAIN_FILE=$(CMAKE_TOOLCHAIN_FILE),) \
		$(CMAKE_ARGS) \
		-DERL_INCLUDE_DIR=$(ERL_INCLUDE_DIR) -DEIGEN_DIR=$(EIGEN_DIR) -DFINE_INCLUDE=$(FINE_INCLUDE) \
		-DNX_EIGEN_FFT_LIB=$(NX_EIGEN_FFT_LIB) \
		$(if $(NX_EIGEN_FFT_SO),-DNX_EIGEN_FFT_SO=$(NX_EIGEN_FFT_SO),) \
		$(if $(MIX_APP_PATH),-DAPP_PRIV=$(MIX_APP_PATH)/priv,)
	$(CMAKE) --build $(CMAKE_BUILD_DIR) --config $(CMAKE_BUILD_TYPE) --parallel
else
	$(CXX) $(CFLAGS) $(FFT_CFLAGS) c_src/nx_eigen_nif.cpp $(FFT_SRCS) $(LDFLAGS) $(FFT_LDFLAGS) -o $(LIB_NAME)
endif

clean:
	rm -rf $(LIB_NAME) $(CMAKE_BUILD_DIR) eigen-$(EIGEN_VERSION)*

# Precompilation targets
precompile:
	@bash scripts/precompile-docker.sh

# Test precompiled binaries
test-precompiled:
	@bash scripts/test-precompiled.sh

test-precompiled-target:
	@if [ -z "$(TARGET)" ]; then \
		echo "Error: TARGET not specified. Usage: make test-precompiled-target TARGET=x86_64-linux-gnu"; \
		exit 1; \
	fi
	@bash scripts/test-precompiled.sh $(TARGET)

.PHONY: all clean check-deps precompile test-precompiled test-precompiled-target

