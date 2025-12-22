defmodule NxEigen.LinAlgTest do
  use ExUnit.Case, async: true
  import Nx.Testing

  setup do
    Nx.default_backend(NxEigen.Backend)
    :ok
  end

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

  describe "cholesky" do
    test "basic cholesky" do
      t = Nx.tensor([[20.0, 17.6], [17.6, 16.0]])
      expected = Nx.tensor([
        [4.4721360206604, 0.0],
        [3.9354796409606934, 0.7155418395996094]
      ])
      assert_all_close(Nx.LinAlg.cholesky(t), expected)
    end

    test "batched cholesky" do
      t = Nx.tensor([[[2.0, 3.0], [3.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]])
      expected = Nx.tensor([
        [
          [1.4142135381698608, 0.0],
          [2.1213204860687256, 0.7071064710617065]
        ],
        [
          [1.0, 0.0],
          [0.0, 1.0]
        ]
      ])
      assert_all_close(Nx.LinAlg.cholesky(t), expected)
    end

    test "larger matrix cholesky" do
      t = Nx.tensor([
        [6.0, 3.0, 4.0, 8.0],
        [3.0, 6.0, 5.0, 1.0],
        [4.0, 5.0, 10.0, 7.0],
        [8.0, 1.0, 7.0, 25.0]
      ])
      expected = Nx.tensor([
        [2.4494898319244385, 0.0, 0.0, 0.0],
        [1.2247447967529297, 2.1213202476501465, 0.0, 0.0],
        [1.6329931020736694, 1.41421377658844, 2.309401035308838, 0.0],
        [3.265986204147339, -1.4142134189605713, 1.5877134799957275, 3.132491111755371]
      ])
      assert_all_close(Nx.LinAlg.cholesky(t), expected)
    end

    test "complex cholesky" do
      t = Nx.tensor([[1.0, Complex.new(0, -2)], [Complex.new(0, 2), 5.0]])
      expected = Nx.tensor([
        [Complex.new(1.0, 0.0), Complex.new(0.0, 0.0)],
        [Complex.new(0.0, 2.0), Complex.new(1.0, 0.0)]
      ])
      # Complex support might vary, checking if it runs and matches
      assert_all_close(Nx.LinAlg.cholesky(t), expected)
    end

    test "vectorized cholesky" do
      t = Nx.tensor([[[2.0, 3.0], [3.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]]) |> Nx.vectorize(x: 2)
      expected = Nx.tensor([
        [
          [1.4142135381698608, 0.0],
          [2.1213204860687256, 0.7071064710617065]
        ],
        [
          [1.0, 0.0],
          [0.0, 1.0]
        ]
      ])
      assert_all_close(Nx.LinAlg.cholesky(t) |> Nx.devectorize(), expected)
    end
  end

  describe "norm" do
    test "vector norms" do
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([3, 4])), Nx.tensor(5.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([3, 4]), ord: 1), Nx.tensor(7.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([3, -4]), ord: :inf), Nx.tensor(4.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([3, -4]), ord: :neg_inf), Nx.tensor(3.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([3, -4, 0, 0]), ord: 0), Nx.tensor(2.0))
    end

    test "matrix norms" do
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[3, -1], [2, -4]]), ord: -1), Nx.tensor(5.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[3, -2], [2, -4]]), ord: 1), Nx.tensor(6.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[3, -2], [2, -4]]), ord: :neg_inf), Nx.tensor(5.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[3, -2], [2, -4]]), ord: :inf), Nx.tensor(6.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[3, 0], [0, -4]]), ord: :frobenius), Nx.tensor(5.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[1, 0, 0], [0, -4, 0], [0, 0, 9]]), ord: :nuclear), Nx.tensor(14.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[1, 0, 0], [0, -4, 0], [0, 0, 9]]), ord: -2), Nx.tensor(1.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[3, 0], [0, -4]])), Nx.tensor(5.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[3, 4], [0, -4]]), axes: [1]), Nx.tensor([5.0, 4.0]))

      # Complex norms
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[Complex.new(0, 3), 4], [4, 0]]), axes: [0]), Nx.tensor([5.0, 4.0]))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[Complex.new(0, 3), 0], [4, 0]]), ord: :neg_inf), Nx.tensor(3.0))
      assert_all_close(Nx.LinAlg.norm(Nx.tensor([[0, 0], [0, 0]])), Nx.tensor(0.0))
    end

    test "norm error cases" do
      assert_raise ArgumentError, fn ->
        Nx.LinAlg.norm(Nx.tensor([3, 4]), ord: :frobenius)
      end
    end
  end

  describe "triangular_solve" do
    test "basic triangular_solve" do
      a = Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
      b = Nx.tensor([4, 2, 4, 2])
      expected = Nx.tensor([1.3333333730697632, -0.6666666865348816, 2.6666667461395264, -1.3333333730697632])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b), expected)
    end

    test "triangular_solve f64" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: :f64)
      b = Nx.tensor([1, 2, 1])
      expected = Nx.tensor([1.0, 1.0, -1.0], type: :f64)
      assert_all_close(Nx.LinAlg.triangular_solve(a, b), expected)
    end

    test "triangular_solve matrix b" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1, 2, 3], [2, 2, 4], [2, 0, 1]])
      expected = Nx.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
      ])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b), expected)
    end

    test "triangular_solve upper" do
      a = Nx.tensor([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 3]])
      b = Nx.tensor([2, 4, 2, 4])
      expected = Nx.tensor([-1.3333333730697632, 2.6666667461395264, -0.6666666865348816, 1.3333333730697632])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b, lower: false), expected)
    end

    test "triangular_solve right side" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      b = Nx.tensor([[0, 2, 1], [1, 1, 0], [3, 3, 1]])
      expected = Nx.tensor([
        [-1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
      ])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b, left_side: false), expected)
    end

    test "triangular_solve transform_a" do
      a = Nx.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], type: :f64)
      b = Nx.tensor([1, 2, 1])
      expected = Nx.tensor([1.0, 1.0, -1.0], type: :f64)
      assert_all_close(Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose, lower: false), expected)

      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: :f64)
      assert_all_close(Nx.LinAlg.triangular_solve(a, b, transform_a: :none), expected)
    end

    test "triangular_solve broadcast" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      b = Nx.tensor([[0, 1, 3], [2, 1, 3]])
      expected = Nx.tensor([
        [2.0, -5.0, 3.0],
        [4.0, -5.0, 3.0]
      ])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b, left_side: false), expected)

      b = Nx.tensor([[0, 2], [3, 0], [0, 0]])
      expected_left = Nx.tensor([
        [0.0, 2.0],
        [3.0, -2.0],
        [-6.0, 2.0]
      ])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b, left_side: true), expected_left)
    end

    test "triangular_solve complex" do
      a = Nx.tensor([
        [1, 0, 0],
        [1, Complex.new(0, 1), 0],
        [Complex.new(0, 1), 1, 1]
      ])
      b = Nx.tensor([1, -1, Complex.new(3, 3)])
      expected = Nx.tensor([Complex.new(1.0, 0.0), Complex.new(0.0, 2.0), Complex.new(3.0, 0.0)])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b), expected)
    end

    test "triangular_solve batch" do
      a = Nx.tensor([[[1, 0], [2, 3]], [[4, 0], [5, 6]]])
      b = Nx.tensor([[2, -1], [3, 7]])
      expected = Nx.tensor([
        [2.0, -1.6666666269302368],
        [0.75, 0.5416666865348816]
      ])
      assert_all_close(Nx.LinAlg.triangular_solve(a, b), expected)
    end

    test "triangular_solve vectorized" do
      a = Nx.tensor([[[1, 1], [0, 1]], [[2, 0], [0, 2]]]) |> Nx.vectorize(x: 2)
      b = Nx.tensor([[[2, 1], [5, -1]]]) |> Nx.vectorize(x: 1, y: 2)
      expected = Nx.tensor([
        [
          [1.0, 1.0],
          [6.0, -1.0]
        ],
        [
          [1.0, 0.5],
          [2.5, -0.5]
        ]
      ])
      # Vectorized assertions might need adjustment depending on how vectorization is handled in tests
      # But assertion on result tensor should work if backend supports it
      result = Nx.LinAlg.triangular_solve(a, b, lower: false) |> Nx.devectorize()
      assert_all_close(result, expected)
    end
  end

  describe "solve" do
    test "basic solve" do
      a = Nx.tensor([[1, 3, 2, 1], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
      b = Nx.tensor([-3, 0, 4, -2])
      expected = Nx.tensor([1.0, -2.0, 3.0, -4.0])
      assert_all_close(Nx.LinAlg.solve(a, b), expected)
    end

    test "solve f64" do
      a = Nx.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1]], type: :f64)
      b = Nx.tensor([0, 2, 1])
      expected = Nx.tensor([1.0, 1.0, -1.0], type: :f64)
      assert_all_close(Nx.LinAlg.solve(a, b), expected)
    end

    test "solve matrix b" do
      a = Nx.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[2, 2, 3], [2, 2, 4], [2, 0, 1]])
      expected = Nx.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
      ])
      assert_all_close(Nx.LinAlg.solve(a, b), expected)
    end

    test "solve batched" do
      a = Nx.tensor([[[14, 10], [9, 9]], [[4, 11], [2, 3]]])
      b = Nx.tensor([[[2, 4], [3, 2]], [[1, 5], [-3, -1]]])
      expected = Nx.tensor([
        [
          [-0.33333343, 0.44444454],
          [0.6666668, -0.22222236]
        ],
        [
          [-3.6, -2.6],
          [1.4, 1.4]
        ]
      ])
      assert_all_close(Nx.LinAlg.solve(a, b), expected, atol: 1.0e-5)
    end

    test "solve vectorized" do
      a = Nx.tensor([[[1, 1], [0, 1]], [[2, 0], [0, 2]]]) |> Nx.vectorize(x: 2)
      b = Nx.tensor([[[2, 1], [5, -1]]]) |> Nx.vectorize(x: 1, y: 2)
      expected = Nx.tensor([
        [
          [1.0, 1.0],
          [6.0, -1.0]
        ],
        [
          [1.0, 0.5],
          [2.5, -0.5]
        ]
      ])
      result = Nx.LinAlg.solve(a, b) |> Nx.devectorize()
      assert_all_close(result, expected)
    end
  end

  describe "qr" do
    test "basic qr" do
      {q, r} = Nx.LinAlg.qr(Nx.tensor([[-3, 2, 1], [0, 1, 1], [0, 0, -1]]))
      expected_q = Nx.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
      ])
      expected_r = Nx.tensor([
        [-3.0, 2.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, -1.0]
      ])
      # QR decomposition is unique up to sign of columns of Q (and rows of R),
      # so direct comparison might be tricky if signs flip, but for this simple case it should match or we check properties
      # For now, let's try direct assertion as in doctest
      assert_all_close(q, expected_q)
      assert_all_close(r, expected_r)
    end

    test "reduced vs complete qr" do
      t = Nx.tensor([[3, 2, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1]], type: :f32)
      {q, r} = Nx.LinAlg.qr(t, mode: :reduced)
      # Shapes should be correct
      assert Nx.shape(q) == {4, 3}
      assert Nx.shape(r) == {3, 3}
      # Reconstruct
      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-5)

      t2 = Nx.tensor([[3, 2, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], type: :f32)
      {q2, r2} = Nx.LinAlg.qr(t2, mode: :complete)
      assert Nx.shape(q2) == {4, 4}
      assert Nx.shape(r2) == {4, 3}
      assert_all_close(Nx.dot(q2, r2), t2, atol: 1.0e-5)
    end
  end

  describe "svd" do
    test "basic svd" do
      {u, s, vt} = Nx.LinAlg.svd(Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
      expected_s = Nx.tensor([1.0, 1.0, 1.0])
      assert_all_close(s, expected_s)
      # Check reconstruction U * S * Vt = A
      # Use underscore to avoid unused variable warning if we were not using them, but we are using them.
      # The warning was because I had a commented out line and then a re-assignment.
      reconstructed = Nx.dot(u, Nx.make_diagonal(s)) |> Nx.dot(vt)
      original = Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], type: :f32)
      assert_all_close(reconstructed, original, atol: 1.0e-5)
    end

    test "svd shapes and values" do
      t = Nx.tensor([[2, 0, 0], [0, 3, 0], [0, 0, -1], [0, 0, 0]])
      {_u, s, _vt} = Nx.LinAlg.svd(t)
      expected_s = Nx.tensor([3.0, 1.9999998807907104, 1.0])
      assert_all_close(s, expected_s)

      # Check full_matrices=false
      {u_red, s_red, vt_red} = Nx.LinAlg.svd(t, full_matrices?: false)
      assert Nx.shape(u_red) == {4, 3}
      assert Nx.shape(s_red) == {3}
      assert Nx.shape(vt_red) == {3, 3}
      assert_all_close(s_red, expected_s)
    end
  end

  describe "lu" do
    test "basic lu" do
      t = Nx.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      {p, l, u} = Nx.LinAlg.lu(t)
      # Check reconstruction P . L . U = A
      # Note: P here is usually a permutation matrix
      reconstructed = p |> Nx.dot(l) |> Nx.dot(u)
      assert_all_close(reconstructed, t, atol: 1.0e-5)
    end

    test "batched lu" do
      t = Nx.tensor([[[9, 8, 7], [6, 5, 4], [3, 2, 1]], [[-1, 0, -1], [1, 0, 1], [1, 1, 1]]])
      {p, l, u} = Nx.LinAlg.lu(t)
      reconstructed = Nx.dot(p, [2], [0], l, [1], [0]) |> Nx.dot([2], [0], u, [1], [0])
      assert_all_close(reconstructed, t, atol: 1.0e-5)
    end
  end

  describe "matrix_power" do
    test "basic matrix_power" do
      t = Nx.tensor([[1, 2], [3, 4]])
      assert_all_close(Nx.LinAlg.matrix_power(t, 0), Nx.eye(2, type: :s32))

      expected_6 = Nx.tensor([[5743, 8370], [12555, 18298]], type: :s32)
      assert_all_close(Nx.LinAlg.matrix_power(t, 6), expected_6)

      expected_inv = Nx.tensor([
        [-2.000000476837158, 1.0000003576278687],
        [1.5000004768371582, -0.5000002384185791]
      ])
      assert_all_close(Nx.LinAlg.matrix_power(t, -1), expected_inv)
    end

    test "batched matrix_power" do
      t = Nx.iota({2, 2, 2})
      result = Nx.LinAlg.matrix_power(t, 3)
      expected = Nx.tensor([
        [
          [6, 11],
          [22, 39]
        ],
        [
          [514, 615],
          [738, 883]
        ]
      ], type: :s32)
      assert_all_close(result, expected)
    end
  end

  describe "determinant" do
    test "basic determinant" do
      assert_all_close(Nx.LinAlg.determinant(Nx.tensor([[1, 2], [3, 4]])), Nx.tensor(-2.0))
      assert_all_close(Nx.LinAlg.determinant(Nx.tensor([[1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [7.0, 8.0, 9.0]])), Nx.tensor(48.0))
      assert_all_close(Nx.LinAlg.determinant(Nx.tensor([[1.0, 0.0], [3.0, 0.0]])), Nx.tensor(0.0))
    end

    test "larger determinant" do
      t = Nx.tensor([
        [1, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 1, -2, 3],
        [0, 7, 8, 9.0]
      ])
      assert_all_close(Nx.LinAlg.determinant(t), Nx.tensor(47.999996185302734))
    end

    test "complex determinant" do
      t = Nx.tensor([[1, 0, 0], [0, Complex.new(0, 2), 0], [0, 0, 3]])
      expected = Nx.tensor(Complex.new(0, 6), type: :c64)
      assert_all_close(Nx.LinAlg.determinant(t), expected)
    end
  end

  describe "pinv" do
    test "pinv scalar and vector" do
      assert_all_close(Nx.LinAlg.pinv(2), Nx.tensor(0.5))
      assert_all_close(Nx.LinAlg.pinv(0), Nx.tensor(0.0))
      assert_all_close(Nx.LinAlg.pinv(Nx.tensor([0, 1, 2])), Nx.tensor([0.0, 0.2, 0.4]))
      assert_all_close(Nx.LinAlg.pinv(Nx.tensor([0, 0, 0])), Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "pinv matrix" do
      t = Nx.tensor([[1, 1], [3, 4]])
      expected = Nx.tensor([
        [4.000002861022949, -1.0000008344650269],
        [-3.000002384185791, 1.0000005960464478]
      ])
      assert_all_close(Nx.LinAlg.pinv(t), expected)

      t2 = Nx.tensor([[0.5, 0], [0, 1], [0.5, 0]])
      expected2 = Nx.tensor([
        [0.9999999403953552, 0.0, 0.9999998807907104],
        [0.0, 1.0, 0.0]
      ])
      assert_all_close(Nx.LinAlg.pinv(t2), expected2)
    end
  end

  describe "least_squares" do
    test "least_squares" do
      assert_all_close(
        Nx.LinAlg.least_squares(Nx.tensor([[1, 2], [2, 3]]), Nx.tensor([1, 2])),
        Nx.tensor([1.0000028610229492, -2.384185791015625e-6]),
        atol: 1.0e-5
      )

      assert_all_close(
        Nx.LinAlg.least_squares(Nx.tensor([[0, 1], [1, 1], [2, 1], [3, 1]]), Nx.tensor([-1, 0.2, 0.9, 2.1])),
        Nx.tensor([0.9999998211860657, -0.9500012993812561]),
        atol: 1.0e-5
      )

      assert_all_close(
        Nx.LinAlg.least_squares(Nx.tensor([[1, 2, 3], [4, 5, 6]]), Nx.tensor([1, 2])),
        Nx.tensor([-0.05555540323257446, 0.1111111044883728, 0.27777770161628723]),
        atol: 1.0e-5
      )
    end
  end
end
