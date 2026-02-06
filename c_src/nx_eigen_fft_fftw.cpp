// nx_eigen_fft_fftw.cpp -- Default FFT implementation using FFTW3
//
// Implements the interface declared in nx_eigen_fft.h.
// Linked into the NIF when NX_EIGEN_FFT_LIB=fftw (the default).

#include "nx_eigen_fft.h"
#include <fftw3.h>

extern "C" {

// Float32 variants
int nx_eigen_fft_forward_f32(const float *in, float *out, int n) {
  fftwf_plan plan = fftwf_plan_dft_1d(
      n,
      const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(in)),
      reinterpret_cast<fftwf_complex *>(out), FFTW_FORWARD, FFTW_ESTIMATE);
  if (!plan)
    return -1;
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  return 0;
}

int nx_eigen_fft_inverse_f32(const float *in, float *out, int n) {
  fftwf_plan plan = fftwf_plan_dft_1d(
      n,
      const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(in)),
      reinterpret_cast<fftwf_complex *>(out), FFTW_BACKWARD, FFTW_ESTIMATE);
  if (!plan)
    return -1;
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  return 0;
}

// Float64 variants
int nx_eigen_fft_forward_f64(const double *in, double *out, int n) {
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

int nx_eigen_fft_inverse_f64(const double *in, double *out, int n) {
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
