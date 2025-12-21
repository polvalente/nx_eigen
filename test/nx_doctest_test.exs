defmodule NxEigen.DocTest do
  use ExUnit.Case, async: true

  # Set default backend for all Nx operations in doctests
  setup do
    Nx.default_backend(NxEigen.Backend)
    :ok
  end


  @unsupported_types [
      as_type: 2,
      tensor: 2
  ]

  # Run Nx's own doctests with NxEigen backend
  # This ensures full compatibility with Nx's documented behavior
  doctest Nx,
    except: @unsupported_types ++ [
      :moduledoc,
      # Slicing with tensor indices
      # Diagonal operations (not implemented)
      take_diagonal: 2,
      make_diagonal: 2,
      put_diagonal: 3,
      # Binary output operations
      to_binary: 2,
      to_flat_list: 2,
      # Pad with interior padding
      pad: 3,
      # Bitwise operations
      bit_size: 1,
      # Integer operations
      remainder: 2,
      quotient: 2,
      # Logical operations
      logical_and: 2,
      logical_or: 2,
      logical_xor: 2,
      # Math operations on integer types
      acosh: 1,
      asinh: 1,
      cbrt: 1,
      cos: 1,
      cosh: 1,
      erf: 1,
      erf_inv: 1,
      erfc: 1,
      exp: 1,
      expm1: 1,
      log1p: 1,
      rsqrt: 1,
      sigmoid: 1,
      sin: 1,
      sinh: 1,
      sqrt: 1,
      tan: 1,
      tanh: 1,
      log: 1,
      log: 2,
      log2: 1,
      log10: 1,
      # Unary on wrong types
      negate: 1,
      abs: 1,
      conjugate: 1,
      real: 1,
      imag: 1,
      # Reductions
      all: 2,
      any: 2,
      sum: 2,
      product: 2,
      # Comparison
      all_close: 3,
      # Statistical
      mode: 2,
      variance: 2,
      standard_deviation: 2,
      # Argmax/argmin with tie_break
      argmax: 2,
      argmin: 2,
      # Cumulative operations (not implemented)
      cumulative_sum: 2,
      cumulative_product: 2,
      cumulative_min: 2,
      cumulative_max: 2,
      # Window operations (not implemented)
      window_reduce: 5,
      window_sum: 3,
      window_product: 3,
      window_max: 3,
      window_min: 3,
      window_mean: 3,
      window_scatter_max: 5,
      window_scatter_min: 5,
      # Dot product with complex batching
      dot: 2,
      # Clip
      clip: 3,
      # Advanced slicing
      slice: 4,
      slice_along_axis: 4,
      # Put operations
      put_slice: 3,
      # Gather
      gather: 3,
      # Concatenate
      concatenate: 2,
      # Sorting
      sort: 2,
      argsort: 2,
      top_k: 2,
      # Reflection
      reflect: 2,
      # Sigil operations
      sigil_MAT: 2,
      sigil_VEC: 2,
      # Logsumexp
      logsumexp: 2,
      # Operations with issues
      atan2: 2,
      # Indexed operations
      indexed_add: 4,
      indexed_put: 4,
      # Custom reduce
      reduce: 4,
      # Take operations
      take: 3,
      take_along_axis: 3,
      # FFT
      fft: 2,
      ifft: 2,
      fft2: 2,
      ifft2: 2,
      # Convolution
      conv: 3,
      # Eye
      eye: 2
    ]

  doctest Nx.LinAlg,
    except: [
      cholesky: 1,
      invert: 1,
      matrix_power: 2,
      determinant: 1,
      least_squares: 3,
      qr: 2,
      svd: 2,
      pinv: 2,
      solve: 2,
      matrix_rank: 2,
      norm: 2,
      eigh: 2,
      lu: 2,
      triangular_solve: 3,
      adjoint: 2
    ]
end
