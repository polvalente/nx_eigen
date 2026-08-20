// nx_eigen_fft_eigen.cpp -- FFT via Eigen's own FFT module (no external library)
//
// Implements the interface declared in nx_eigen_fft.h.
// Linked into the NIF when NX_EIGEN_FFT_LIB=eigen.
//
// Eigen's FFT module defaults to its vendored kissfft implementation, which is
// MPL-2.0 like the rest of the Eigen we use, so this backend adds no
// dependency beyond the Eigen headers the NIF already compiles against.
//
// Three details of Eigen::FFT matter here:
//
//   * `inv` scales by 1/n unless the Unscaled flag is set, while this
//     interface is unnormalised in both directions — the caller divides.
//   * The plan cache lives on the FFT object and isn't thread safe, so each
//     scheduler thread gets its own via thread_local.
//   * kissfft only has butterflies for radix 2/3/4/5. Any larger prime factor
//     falls back to a generic butterfly costing O(n * p), which for a prime
//     length is O(n²) — seconds for a few tens of thousands of samples. Those
//     lengths go through Bluestein's algorithm instead (see below).

#include "nx_eigen_fft.h"

#include <unsupported/Eigen/FFT>

#include <cmath>
#include <complex>
#include <vector>

namespace {

constexpr double pi = 3.14159265358979323846;

// Sign of the exponent: forward transforms use e^-i…, inverse e^+i….
enum Direction { kForward = -1, kInverse = 1 };

template <typename T> Eigen::FFT<T> &fft_for_thread() {
  thread_local Eigen::FFT<T> fft = [] {
    Eigen::FFT<T> f;
    f.SetFlag(Eigen::FFT<T>::Unscaled);
    return f;
  }();

  return fft;
}

int largest_prime_factor(int n) {
  int largest = 1;

  for (int p = 2; p <= n / p; ++p) {
    while (n % p == 0) {
      largest = p;
      n /= p;
    }
  }

  return n > 1 ? n : largest;
}

// kissfft's generic butterfly costs about n*p for largest prime factor p, while
// Bluestein costs three power-of-two transforms of m >= 2n-1, about 6*n*log2(m).
// The two cross over around p == 6*log2(2n), which is in the sixties for the
// lengths this is likely to see, so that's the threshold. Below it the direct
// transform is cheaper and exact; above it the generic butterfly is the one
// that turns into seconds.
bool prefer_bluestein(int n) { return largest_prime_factor(n) > 64; }

// exp(direction * i * pi * k^2 / n), with k^2 reduced mod 2n first so the
// angle stays small enough for cos/sin to keep their precision at large k.
template <typename T> std::complex<T> chirp(int k, int n, int direction) {
  const int64_t modulus = 2LL * n;
  const int64_t reduced = static_cast<int64_t>(k) % modulus;
  const int64_t k_squared = reduced * reduced % modulus;
  const double angle = direction * pi * static_cast<double>(k_squared) / n;

  return {static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle))};
}

// Bluestein's algorithm: rewrite the DFT as a convolution, which is then done
// with power-of-two transforms kissfft handles well. Uses
// jk = (j² + k² - (k-j)²)/2 to split the kernel into two chirps and one
// convolution.
template <typename T>
void bluestein(const std::complex<T> *in, std::complex<T> *out, int n,
               int direction) {
  int m = 1;
  while (m < 2 * n - 1)
    m <<= 1;

  std::vector<std::complex<T>> a(m, std::complex<T>(0, 0));
  std::vector<std::complex<T>> b(m, std::complex<T>(0, 0));

  for (int j = 0; j < n; ++j)
    a[j] = in[j] * chirp<T>(j, n, direction);

  // The convolution kernel is symmetric about m, so the negative lags wrap to
  // the top of the buffer.
  b[0] = chirp<T>(0, n, -direction);
  for (int j = 1; j < n; ++j) {
    const std::complex<T> value = chirp<T>(j, n, -direction);
    b[j] = value;
    b[m - j] = value;
  }

  auto &fft = fft_for_thread<T>();
  std::vector<std::complex<T>> fa(m), fb(m), product(m), convolved(m);

  fft.fwd(fa.data(), a.data(), m);
  fft.fwd(fb.data(), b.data(), m);

  for (int i = 0; i < m; ++i)
    product[i] = fa[i] * fb[i];

  // Unscaled, so the 1/m the circular convolution needs is applied below.
  fft.inv(convolved.data(), product.data(), m);

  const T scale = T(1) / static_cast<T>(m);
  for (int k = 0; k < n; ++k)
    out[k] = convolved[k] * scale * chirp<T>(k, n, direction);
}

// Eigen writes the transform into `dst` while reading `src`, so an aliased
// in-place call needs its input copied out of the way first. The interface
// permits aliasing, so always transform from a scratch copy.
template <typename T>
int transform(const T *in, T *out, int n, int direction) {
  if (n <= 0)
    return -1;

  const auto *in_complex = reinterpret_cast<const std::complex<T> *>(in);
  auto *out_complex = reinterpret_cast<std::complex<T> *>(out);

  // kissfft's factorisation has nothing to do for a single sample.
  if (n == 1) {
    out_complex[0] = in_complex[0];
    return 0;
  }

  thread_local std::vector<std::complex<T>> scratch;
  scratch.assign(in_complex, in_complex + n);

  if (prefer_bluestein(n)) {
    bluestein<T>(scratch.data(), out_complex, n, direction);
  } else if (direction == kForward) {
    fft_for_thread<T>().fwd(out_complex, scratch.data(), n);
  } else {
    fft_for_thread<T>().inv(out_complex, scratch.data(), n);
  }

  return 0;
}

} // namespace

extern "C" {

int nx_eigen_fft_forward_f32(const float *in, float *out, int n) {
  return transform(in, out, n, kForward);
}

int nx_eigen_fft_inverse_f32(const float *in, float *out, int n) {
  return transform(in, out, n, kInverse);
}

int nx_eigen_fft_forward_f64(const double *in, double *out, int n) {
  return transform(in, out, n, kForward);
}

int nx_eigen_fft_inverse_f64(const double *in, double *out, int n) {
  return transform(in, out, n, kInverse);
}

} // extern "C"
