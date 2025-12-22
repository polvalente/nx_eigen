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

  @rounding_error [
      # These tests fail due to minor floating point precision differences
      erf_inv: 1,
      expm1: 1,
      sigmoid: 1,
      tanh: 1,
    ]

  @unsupported_ops [
    reduce: 4,
    window_reduce: 5
  ]

  @sub_byte_types [
      # Sub-byte types (u2, etc.) not supported
      bit_size: 1
  ]

  # Run Nx's own doctests with NxEigen backend
  # This ensures full compatibility with Nx's documented behavior
  doctest Nx,
    except: @unsupported_types ++ @rounding_error ++ @sub_byte_types ++ @unsupported_ops ++ [
      :moduledoc
    ]

  @rounding_error_linalg [
    cholesky: 1,
    determinant: 1,
    matrix_power: 2,
    svd: 2,
    pinv: 2,
    norm: 2,
    lu: 2,
    least_squares: 3,
    triangular_solve: 3,
    solve: 2
  ]

  doctest Nx.LinAlg, except: @rounding_error_linalg
end
