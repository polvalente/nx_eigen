// nx_eigen_fft_none.cpp -- Stub FFT implementation (FFT disabled)
//
// Implements the interface declared in nx_eigen_fft.h.
// Linked into the NIF when NX_EIGEN_FFT_LIB=none.
// Every call returns -1 (failure); the NIF translates this into
// a runtime error with a descriptive message.

#include "nx_eigen_fft.h"

extern "C" {

int nx_eigen_fft_forward_f32(const float * /*in*/, float * /*out*/, int /*n*/) {
  return -1;
}

int nx_eigen_fft_inverse_f32(const float * /*in*/, float * /*out*/, int /*n*/) {
  return -1;
}

int nx_eigen_fft_forward_f64(const double * /*in*/, double * /*out*/, int /*n*/) {
  return -1;
}

int nx_eigen_fft_inverse_f64(const double * /*in*/, double * /*out*/, int /*n*/) {
  return -1;
}

} // extern "C"
