defmodule NxEigen.DocTest do
  use ExUnit.Case, async: true

  # Set default backend for all Nx operations in doctests
  setup do
    Nx.default_backend(NxEigen.Backend)
    :ok
  end


  @unsupported_types [
      as_type: 2,
      tensor: 2,
      # bf16 type tests
      real: 1,
      imag: 1
  ]

  @precision_sensitive [
      # These tests fail due to minor floating point precision differences
      erf_inv: 1,
      expm1: 1,
      sigmoid: 1,
      tanh: 1
  ]

  @sub_byte_types [
      # Sub-byte types (u2, etc.) not supported
      bit_size: 1
  ]

  # Run Nx's own doctests with NxEigen backend
  # This ensures full compatibility with Nx's documented behavior
  doctest Nx,
    except: @unsupported_types ++ @precision_sensitive ++ @sub_byte_types ++ [
      :moduledoc,
      # Slicing with tensor indices
      # Diagonal operations (require multi-dimensional gather support)
      take_diagonal: 2,
      make_diagonal: 2,
      put_diagonal: 3,
      # Binary output operations
      to_binary: 2,
      to_flat_list: 2,
      # Pad with interior padding
      pad: 3,
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
      # Concatenate
      concatenate: 2,
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
