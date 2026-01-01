cmake_minimum_required(VERSION 3.20)

project(nx_eigen_nif LANGUAGES CXX)

# Disable FFTW-dependent NIFs (fft/ifft). When ON, we don't look for fftw3 at all.
option(NX_EIGEN_DISABLE_FFTW "Disable FFTW-dependent NIFs (fft/ifft)" OFF)

# Paths are passed in from the Makefile (or can be set manually).
set(ERL_INCLUDE_DIR "" CACHE PATH "Path to Erlang/ERTS include directory (contains erl_nif.h)")
set(EIGEN_DIR "" CACHE PATH "Path to Eigen include directory (contains Eigen/)")
set(FINE_INCLUDE "" CACHE PATH "Path to Fine C include directory (contains fine.hpp)")
set(FFTW_DIR "" CACHE PATH "Prefix for FFTW installation (expects include/ and lib/)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nx_eigen MODULE c_src/nx_eigen_nif.cpp)
set_target_properties(nx_eigen PROPERTIES
  OUTPUT_NAME "nx_eigen"   # -> libnx_eigen.*
  PREFIX "lib"
  SUFFIX ".so"            # Erlang loads NIFs as .so even on macOS
)

# Always place the built NIF exactly where :code.priv_dir expects it.
set_target_properties(nx_eigen PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/priv"
)

target_compile_options(nx_eigen PRIVATE -O3)
target_compile_definitions(nx_eigen PRIVATE EIGEN_MPL2_ONLY)
set_target_properties(nx_eigen PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ERL_INCLUDE_DIR STREQUAL "")
  message(FATAL_ERROR "ERL_INCLUDE_DIR is required (set it to your erts-*/include directory).")
endif()
if(EIGEN_DIR STREQUAL "")
  message(FATAL_ERROR "EIGEN_DIR is required (set it to your Eigen root include directory).")
endif()
if(FINE_INCLUDE STREQUAL "")
  message(FATAL_ERROR "FINE_INCLUDE is required (set it to deps/fine/c_include).")
endif()

target_include_directories(nx_eigen PRIVATE
  "${ERL_INCLUDE_DIR}"
  "${EIGEN_DIR}"
  "${FINE_INCLUDE}"
)

if(NX_EIGEN_DISABLE_FFTW)
  target_compile_definitions(nx_eigen PRIVATE NX_EIGEN_DISABLE_FFTW)
else()
  # FFTW: prefer an explicit FFTW_DIR (best for cross-compilation), otherwise fall back to system search.
  if(NOT FFTW_DIR STREQUAL "")
    find_path(FFTW3_INCLUDE_DIR fftw3.h
      PATHS "${FFTW_DIR}/include"
      NO_DEFAULT_PATH
    )
    find_library(FFTW3_LIBRARY fftw3
      PATHS "${FFTW_DIR}/lib"
      NO_DEFAULT_PATH
    )
  else()
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    find_library(FFTW3_LIBRARY fftw3)
  endif()

  if(FFTW3_INCLUDE_DIR)
    target_include_directories(nx_eigen PRIVATE "${FFTW3_INCLUDE_DIR}")
  endif()
  if(FFTW3_LIBRARY)
    target_link_libraries(nx_eigen PRIVATE "${FFTW3_LIBRARY}")
  else()
    message(FATAL_ERROR "FFTW3 library not found. Set FFTW_DIR, or enable NX_EIGEN_DISABLE_FFTW=ON to build without fft/ifft.")
  endif()
endif()

# NIFs on macOS typically use dynamic lookup instead of linking to libErlang.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_options(nx_eigen PRIVATE "-undefined" "dynamic_lookup")
endif()


