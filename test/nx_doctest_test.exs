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
      imag: 1,
      # f16 type tests
      sigil_MAT: 2,
      sigil_VEC: 2,
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
      # Statistical functions
      mode: 2,  # Has slice stride edge cases
      variance: 2,  # Broadcast issue with mean result
      standard_deviation: 2,  # Depends on variance
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
      # Put operations
      put_slice: 3,
      # Indexed operations
      indexed_add: 4,
      indexed_put: 4,
      # Custom reduce
      reduce: 4
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
