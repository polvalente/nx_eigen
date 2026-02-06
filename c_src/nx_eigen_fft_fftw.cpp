// nx_eigen_fft_fftw.cpp -- Default FFT implementation using FFTW3
//
// Implements the interface declared in nx_eigen_fft.h.
// Linked into the NIF when NX_EIGEN_FFT_LIB=fftw (the default).

#include "nx_eigen_fft.h"
#include <fftw3.h>

extern "C" {

int nx_eigen_fft_forward(const double *in, double *out, int n) {
  fftw_plan plan = fftw_plan_dft_1d(
      n,
      const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(in)),
      reinterpret_cast<fftw_complex *>(out), FFTW_FORWARD, FFTW_ESTIMATE);
  if (!plan)
    return -1;
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  return 0;
}

int nx_eigen_fft_inverse(const double *in, double *out, int n) {
  fftw_plan plan = fftw_plan_dft_1d(
      n,
      const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(in)),
      reinterpret_cast<fftw_complex *>(out), FFTW_BACKWARD, FFTW_ESTIMATE);
  if (!plan)
    return -1;
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  return 0;
}

} // extern "C"
