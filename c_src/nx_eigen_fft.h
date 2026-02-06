// nx_eigen_fft.h -- Pluggable FFT interface for NxEigen
//
// NxEigen calls these two functions for all FFT/IFFT work.  The default
// build links the FFTW3 implementation (nx_eigen_fft_fftw.cpp).  When
// cross-compiling for a target that doesn't have FFTW, you can either:
//
//   1. Set  NX_EIGEN_FFT_LIB=none    → stubs that return an error code.
//   2. Set  NX_EIGEN_FFT_SO=/path/to/your_fft.so  → your own shared
//      library that exports these two symbols with C linkage.
//
// ─── Contract ────────────────────────────────────────────────────────
//
// Both functions receive buffers of interleaved complex doubles:
//
//     [re_0, im_0, re_1, im_1, ..., re_{n-1}, im_{n-1}]
//
// so the total buffer size is  2 * n  doubles.
//
// • nx_eigen_fft_forward  performs the DFT   (forward, unnormalised).
// • nx_eigen_fft_inverse  performs the IDFT  (backward, unnormalised).
//
// The caller normalises the inverse transform by dividing by n.
//
// Return 0 on success, any non-zero value on failure.
// ─────────────────────────────────────────────────────────────────────

#ifndef NX_EIGEN_FFT_H
#define NX_EIGEN_FFT_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward (DFT) 1-D complex-to-complex transform.
//
// @param in   Input: n interleaved complex doubles (2*n doubles total).
// @param out  Output: n interleaved complex doubles (2*n doubles total).
//             May alias `in` (in-place is allowed).
// @param n    Number of complex samples (FFT length).
// @return     0 on success, non-zero on failure.
int nx_eigen_fft_forward(const double *in, double *out, int n);

// Inverse (IDFT) 1-D complex-to-complex transform (unnormalised).
//
// @param in   Input: n interleaved complex doubles (2*n doubles total).
// @param out  Output: n interleaved complex doubles (2*n doubles total).
//             May alias `in` (in-place is allowed).
// @param n    Number of complex samples (FFT length).
// @return     0 on success, non-zero on failure.
int nx_eigen_fft_inverse(const double *in, double *out, int n);

#ifdef __cplusplus
}
#endif

#endif // NX_EIGEN_FFT_H
