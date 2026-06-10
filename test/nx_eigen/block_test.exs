defmodule NxEigen.BlockTest do
  use ExUnit.Case, async: true

  # Nx 0.12 routes a family of ops through the block mechanism — the
  # backend callback block/4: cumulative ops, the LinAlg decompositions,
  # take/take_along_axis, top_k, and friends. Each call site hands the
  # backend a default implementation to fall back on; a backend without
  # block/4 raises UndefinedFunctionError on every one of these ops.

  defp eigen(data, opts \\ []) do
    Nx.tensor(data, opts ++ [backend: NxEigen.Backend])
  end

  defp reference(data, opts \\ []) do
    Nx.tensor(data, opts ++ [backend: Nx.BinaryBackend])
  end

  defp assert_close(result, expected) do
    result = Nx.backend_copy(result, Nx.BinaryBackend)

    assert Nx.to_number(Nx.all_close(result, expected, atol: 1.0e-9)) == 1,
           "expected #{inspect(result)} to match #{inspect(expected)}"
  end

  describe "cumulative blocks" do
    test "cumulative_sum matches the reference backend" do
      data = [1.5, -2.0, 3.25, 0.0, 4.5]

      assert_close(
        Nx.cumulative_sum(eigen(data, type: :f64)),
        Nx.cumulative_sum(reference(data, type: :f64))
      )
    end

    test "cumulative_sum supports axis and reverse on 2D tensors" do
      data = [[1, 2, 3], [4, 5, 6]]

      for axis <- [0, 1], reverse <- [false, true] do
        assert_close(
          Nx.cumulative_sum(eigen(data, type: :s64), axis: axis, reverse: reverse),
          Nx.cumulative_sum(reference(data, type: :s64), axis: axis, reverse: reverse)
        )
      end
    end

    test "cumulative_min, _max, and _product match the reference backend" do
      data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]

      for op <- [&Nx.cumulative_min/1, &Nx.cumulative_max/1, &Nx.cumulative_product/1] do
        assert_close(op.(eigen(data, type: :f64)), op.(reference(data, type: :f64)))
      end
    end
  end

  describe "linalg blocks" do
    test "cholesky factors a positive-definite matrix" do
      a = [[4.0, 2.0], [2.0, 3.0]]
      l = Nx.LinAlg.cholesky(eigen(a, type: :f64))

      # L is lower-triangular and L * Lt reconstructs A.
      assert_close(Nx.dot(l, Nx.transpose(l)), reference(a, type: :f64))
      assert_close(l, Nx.LinAlg.cholesky(reference(a, type: :f64)))
    end

    test "solve matches the reference backend" do
      a = [[3.0, 1.0], [1.0, 2.0]]
      b = [9.0, 8.0]

      assert_close(
        Nx.LinAlg.solve(eigen(a, type: :f64), eigen(b, type: :f64)),
        Nx.LinAlg.solve(reference(a, type: :f64), reference(b, type: :f64))
      )
    end
  end

  describe "gather blocks" do
    test "take_along_axis matches the reference backend" do
      data = [[10, 20, 30], [60, 50, 40]]
      idx = [[2, 1, 0], [0, 2, 1]]

      assert_close(
        Nx.take_along_axis(eigen(data, type: :s64), eigen(idx, type: :s64), axis: 1),
        Nx.take_along_axis(reference(data, type: :s64), reference(idx, type: :s64), axis: 1)
      )
    end

    test "top_k matches the reference backend" do
      data = [4.0, 1.0, 3.0, 2.0]

      {values, indices} = Nx.top_k(eigen(data, type: :f64), k: 2)
      {ref_values, ref_indices} = Nx.top_k(reference(data, type: :f64), k: 2)

      assert_close(values, ref_values)
      assert_close(indices, ref_indices)
    end
  end
end
