ERL_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]), halt().')

# Eigen configuration
# Set EIGEN_DIR to use a local installation, otherwise download automatically
EIGEN_VERSION = 3.4.0
EIGEN_DIR ?= $(CURDIR)/eigen-$(EIGEN_VERSION)
EIGEN_INCLUDE = $(EIGEN_DIR)
FINE_INCLUDE = $(CURDIR)/deps/fine/c_include

# FFTW3 configuration
# 1. If FFTW_DIR is set, use that local installation
# 2. Try to detect system FFTW via pkg-config
# 3. Try Homebrew on macOS
# 4. Otherwise, download and build locally
FFTW_VERSION = 3.3.10
FFTW_SOURCE_DIR = $(CURDIR)/fftw-$(FFTW_VERSION)
FFTW_INSTALL_DIR = $(CURDIR)/fftw-$(FFTW_VERSION)-install

ifndef FFTW_DIR
  # Try pkg-config first
  FFTW_INCLUDE := $(shell pkg-config --cflags fftw3 2>/dev/null)
  FFTW_LIB := $(shell pkg-config --libs fftw3 2>/dev/null)

  # If pkg-config fails, try Homebrew on macOS
  ifeq ($(FFTW_LIB),)
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
      BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
      ifneq ($(BREW_PREFIX),)
        ifneq ($(wildcard $(BREW_PREFIX)/include/fftw3.h),)
          FFTW_INCLUDE = -I$(BREW_PREFIX)/include
          FFTW_LIB = -L$(BREW_PREFIX)/lib -lfftw3
        endif
      endif
    endif
  endif

  # If still not found, use local build
  ifeq ($(FFTW_LIB),)
    FFTW_DIR = $(FFTW_INSTALL_DIR)
    FFTW_INCLUDE = -I$(FFTW_DIR)/include
    FFTW_LIB = -L$(FFTW_DIR)/lib -lfftw3
  endif
else
  FFTW_INCLUDE = -I$(FFTW_DIR)/include
  FFTW_LIB = -L$(FFTW_DIR)/lib -lfftw3
endif

CFLAGS = -fPIC -I$(ERL_INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(FINE_INCLUDE) $(FFTW_INCLUDE) -O3 -std=c++17
LDFLAGS = -shared $(FFTW_LIB)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -undefined dynamic_lookup
endif

LIB_NAME = priv/libnx_eigen.so

all: check-deps priv $(LIB_NAME)

# Check dependencies without rebuilding
check-deps:
	@if [ ! -d "$(EIGEN_DIR)" ]; then \
		echo "Downloading Eigen $(EIGEN_VERSION)..."; \
		curl -L -k https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/eigen-$(EIGEN_VERSION).tar.gz | tar xz || \
		(echo "Failed to download Eigen. Please install manually or set EIGEN_DIR=/path/to/eigen"; exit 1); \
	else \
		echo "Using Eigen from $(EIGEN_DIR)"; \
	fi
	@if [ "$(FFTW_DIR)" = "$(FFTW_INSTALL_DIR)" ] && [ ! -d "$(FFTW_INSTALL_DIR)" ]; then \
		echo "Downloading and building FFTW $(FFTW_VERSION) locally..."; \
		curl -L http://www.fftw.org/fftw-$(FFTW_VERSION).tar.gz | tar xz && \
		cd $(FFTW_SOURCE_DIR) && \
		./configure --prefix=$(FFTW_INSTALL_DIR) --enable-shared --disable-fortran && \
		$(MAKE) && \
		$(MAKE) install; \
	elif [ -n "$(FFTW_LIB)" ]; then \
		echo "Using system FFTW"; \
	else \
		echo "Using FFTW from $(FFTW_DIR)"; \
	fi

priv:
	@mkdir -p priv

$(LIB_NAME): c_src/nx_eigen_nif.cpp | check-deps priv
	$(CXX) $(CFLAGS) $(LDFLAGS) c_src/nx_eigen_nif.cpp -o $(LIB_NAME)

clean:
	rm -rf priv $(LIB_NAME) eigen-$(EIGEN_VERSION)* fftw-$(FFTW_VERSION) fftw-$(FFTW_VERSION)-install

.PHONY: all clean check-deps

