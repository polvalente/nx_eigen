cmake_minimum_required(VERSION 3.20)

project(nx_eigen_nif LANGUAGES CXX)

# Paths are passed in from the Makefile (or can be set manually).
set(ERL_INCLUDE_DIR "" CACHE PATH "Path to Erlang/ERTS include directory (contains erl_nif.h)")
set(EIGEN_DIR "" CACHE PATH "Path to Eigen include directory (contains Eigen/)")
set(FINE_INCLUDE "" CACHE PATH "Path to Fine C include directory (contains fine.hpp)")

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

# FFT library choice.
# Set NX_EIGEN_FFT_LIB to "fftw" (default) or "none" (disables FFT, useful for cross-compilation).
set(NX_EIGEN_FFT_LIB "fftw" CACHE STRING "FFT library to use: fftw or none")

if(NX_EIGEN_FFT_LIB STREQUAL "none")
  target_compile_definitions(nx_eigen PRIVATE NX_EIGEN_DISABLE_FFTW)
  message(STATUS "FFT support: disabled (NX_EIGEN_FFT_LIB=none)")
elseif(NX_EIGEN_FFT_LIB STREQUAL "fftw")
  # Try pkg-config first, fall back to raw -lfftw3
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
  endif()

  if(FFTW3_FOUND)
    target_include_directories(nx_eigen PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(nx_eigen PRIVATE ${FFTW3_LIBRARIES})
    target_link_directories(nx_eigen PRIVATE ${FFTW3_LIBRARY_DIRS})
    message(STATUS "FFT support: FFTW3 (via pkg-config)")
  else()
    # Fall back: just link -lfftw3 and hope the system paths cover it
    target_link_libraries(nx_eigen PRIVATE fftw3)
    message(STATUS "FFT support: FFTW3 (direct link)")
  endif()
else()
  message(FATAL_ERROR "Unsupported NX_EIGEN_FFT_LIB value: ${NX_EIGEN_FFT_LIB}. Use \"fftw\" or \"none\".")
endif()

# NIFs on macOS typically use dynamic lookup instead of linking to libErlang.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_options(nx_eigen PRIVATE "-undefined" "dynamic_lookup")
endif()


