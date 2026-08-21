cmake_minimum_required(VERSION 3.20)

project(nx_eigen_nif LANGUAGES CXX)

# Paths are passed in from the Makefile (or can be set manually).
set(ERL_INCLUDE_DIR "" CACHE PATH "Path to Erlang/ERTS include directory (contains erl_nif.h)")
set(EIGEN_DIR "" CACHE PATH "Path to Eigen include directory (contains Eigen/)")
set(FINE_INCLUDE "" CACHE PATH "Path to Fine C include directory (contains fine.hpp)")
# Output directory for the NIF .so (passed as MIX_APP_PATH/priv by elixir_make)
set(APP_PRIV "" CACHE PATH "Output priv directory (set by elixir_make via MIX_APP_PATH)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable ccache if available to speed up rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "Using ccache for compilation")
endif()

add_library(nx_eigen MODULE c_src/nx_eigen_nif.cpp)
set_target_properties(nx_eigen PROPERTIES
  OUTPUT_NAME "nx_eigen"   # -> libnx_eigen.*
  PREFIX "lib"
  SUFFIX ".so"            # Erlang loads NIFs as .so even on macOS
)

# Place the built NIF where elixir_make/cc_precompiler expect it.
# APP_PRIV is passed by the Makefile when MIX_APP_PATH is set (elixir_make sets it).
# Fall back to ${CMAKE_SOURCE_DIR}/priv for standalone cmake invocations.
if(NOT APP_PRIV STREQUAL "")
  set(NIF_OUTPUT_DIR "${APP_PRIV}")
else()
  set(NIF_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/priv")
endif()
set_target_properties(nx_eigen PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${NIF_OUTPUT_DIR}"
)

target_compile_options(nx_eigen PRIVATE -O3)
target_compile_definitions(nx_eigen PRIVATE EIGEN_MPL2_ONLY)
set_target_properties(nx_eigen PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Strip symbols in Release builds to reduce binary size
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_link_options(nx_eigen PRIVATE -s)
endif()

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

# Use precompiled headers for Eigen to speed up compilation
# Eigen is header-only and template-heavy, so PCH helps significantly
target_precompile_headers(nx_eigen PRIVATE
  <Eigen/Core>
  <Eigen/Dense>
  <complex>
  <cstring>
  <vector>
)

# FFT library choice.
#
# The NIF calls a pluggable C interface (c_src/nx_eigen_fft.h).
# NX_EIGEN_FFT_LIB selects the built-in implementation:
#   "fftw" (default) - compile nx_eigen_fft_fftw.cpp, link FFTW3
#   "none"           - compile nx_eigen_fft_none.cpp  (stubs that return errors)
#
# NX_EIGEN_FFT_SO overrides NX_EIGEN_FFT_LIB: set it to the absolute path of a
# shared library that exports nx_eigen_fft_forward / nx_eigen_fft_inverse.
set(NX_EIGEN_FFT_LIB "fftw" CACHE STRING "FFT implementation: fftw or none")
set(NX_EIGEN_FFT_SO  ""     CACHE FILEPATH "Path to custom FFT .so (overrides NX_EIGEN_FFT_LIB)")

target_include_directories(nx_eigen PRIVATE "${CMAKE_SOURCE_DIR}/c_src")

if(NOT NX_EIGEN_FFT_SO STREQUAL "")
  # ── Custom shared library ──────────────────────────────────────────
  target_link_libraries(nx_eigen PRIVATE "${NX_EIGEN_FFT_SO}")
  set_target_properties(nx_eigen PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
  message(STATUS "FFT support: custom .so (${NX_EIGEN_FFT_SO})")

elseif(NX_EIGEN_FFT_LIB STREQUAL "fftw")
  # ── Default FFTW3 implementation ───────────────────────────────────
  target_sources(nx_eigen PRIVATE c_src/nx_eigen_fft_fftw.cpp)

  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
    pkg_check_modules(FFTW3F QUIET fftw3f)
  endif()

  if(FFTW3_FOUND AND FFTW3F_FOUND)
    target_include_directories(nx_eigen PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(nx_eigen PRIVATE ${FFTW3_LIBRARIES} ${FFTW3F_LIBRARIES})
    message(STATUS "FFT support: FFTW3 dynamic (via pkg-config)")
  else()
    target_link_libraries(nx_eigen PRIVATE fftw3 fftw3f)
    message(STATUS "FFT support: FFTW3 (direct link)")
  endif()

elseif(NX_EIGEN_FFT_LIB STREQUAL "eigen")
  # ── Eigen's own FFT module (no external library) ───────────────────
  target_sources(nx_eigen PRIVATE c_src/nx_eigen_fft_eigen.cpp)
  message(STATUS "FFT support: Eigen FFT module (no external library)")

elseif(NX_EIGEN_FFT_LIB STREQUAL "none")
  # ── Stubs (FFT disabled) ───────────────────────────────────────────
  target_sources(nx_eigen PRIVATE c_src/nx_eigen_fft_none.cpp)
  message(STATUS "FFT support: disabled (NX_EIGEN_FFT_LIB=none)")

else()
  message(FATAL_ERROR "Unsupported NX_EIGEN_FFT_LIB value: ${NX_EIGEN_FFT_LIB}. Use \"fftw\", \"eigen\", \"none\", or set NX_EIGEN_FFT_SO.")
endif()

# NIFs on macOS typically use dynamic lookup instead of linking to libErlang.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  target_link_options(nx_eigen PRIVATE "-undefined" "dynamic_lookup")
endif()


