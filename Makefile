ERL_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]), halt().')

EIGEN_VERSION = 3.4.0
EIGEN_DIR = $(CURDIR)/eigen-$(EIGEN_VERSION)
EIGEN_INCLUDE = $(EIGEN_DIR)
FINE_INCLUDE = $(CURDIR)/deps/fine/c_include

CFLAGS = -fPIC -I$(ERL_INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(FINE_INCLUDE) -O3 -std=c++17
LDFLAGS = -shared

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -undefined dynamic_lookup
endif

LIB_NAME = priv/libnx_eigen.so

all: $(EIGEN_DIR) priv $(LIB_NAME)

$(EIGEN_DIR):
	curl -L https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/eigen-$(EIGEN_VERSION).tar.gz | tar xz

priv:
	mkdir -p priv

$(LIB_NAME): c_src/nx_eigen_nif.cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) c_src/nx_eigen_nif.cpp -o $(LIB_NAME)

clean:
	rm -rf priv $(LIB_NAME) eigen-$(EIGEN_VERSION)*

.PHONY: all clean

