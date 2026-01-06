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

# FFTW is intentionally disabled/removed for now.
CFLAGS = -fPIC -I$(ERL_INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(FINE_INCLUDE) -O3 -std=c++17
LDFLAGS = -shared

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

$(LIB_NAME): c_src/nx_eigen_nif.cpp | check-deps priv
ifeq ($(USE_CMAKE),1)
	$(CMAKE) -S $(CURDIR) -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		$(if $(CMAKE_TOOLCHAIN_FILE),-DCMAKE_TOOLCHAIN_FILE=$(CMAKE_TOOLCHAIN_FILE),) \
		$(CMAKE_ARGS) \
		-DERL_INCLUDE_DIR=$(ERL_INCLUDE_DIR) -DEIGEN_DIR=$(EIGEN_DIR) -DFINE_INCLUDE=$(FINE_INCLUDE)
	$(CMAKE) --build $(CMAKE_BUILD_DIR) --config $(CMAKE_BUILD_TYPE)
else
	$(CXX) $(CFLAGS) $(LDFLAGS) c_src/nx_eigen_nif.cpp -o $(LIB_NAME)
endif

clean:
	rm -rf priv $(LIB_NAME) $(CMAKE_BUILD_DIR) eigen-$(EIGEN_VERSION)*

.PHONY: all clean check-deps

