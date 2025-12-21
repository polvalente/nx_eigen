defmodule NxEigenTest do
  use ExUnit.Case
  doctest NxEigen

  test "basic addition with NxEigen backend" do
    a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32}, backend: NxEigen.Backend)
    b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]], type: {:f, 32}, backend: NxEigen.Backend)

    result = Nx.add(a, b)

    assert result.data.__struct__ == NxEigen.Backend
    assert Nx.to_binary(result) == Nx.to_binary(Nx.tensor([[6.0, 8.0], [10.0, 12.0]], type: {:f, 32}))
  end

  test "subtraction and multiplication with NxEigen backend" do
    a = NxEigen.tensor([[10.0, 20.0]], type: {:f, 32})
    b = NxEigen.tensor([[2.0, 4.0]], type: {:f, 32})

    assert Nx.to_binary(Nx.subtract(a, b)) == Nx.to_binary(Nx.tensor([[8.0, 16.0]], type: {:f, 32}))
    assert Nx.to_binary(Nx.multiply(a, b)) == Nx.to_binary(Nx.tensor([[20.0, 80.0]], type: {:f, 32}))
  end

  test "NxEigen.tensor helper" do
    t = NxEigen.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
    assert t.data.__struct__ == NxEigen.Backend
  end

  test "unary ops" do
    t = NxEigen.tensor([[1.0, 2.0]], type: {:f, 32})

    # exp
    res = Nx.exp(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[:math.exp(1.0), :math.exp(2.0)]], type: {:f, 32}))

    # sqrt
    res = Nx.sqrt(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[:math.sqrt(1.0), :math.sqrt(2.0)]], type: {:f, 32}))
  end

  test "inspect" do
    t = NxEigen.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
    output = inspect(t)
    assert output =~ "Nx.Tensor"
    assert output =~ "1.0"
    assert output =~ "4.0"
  end

  test "comparisons" do
    a = NxEigen.tensor([[1.0, 5.0]], type: {:f, 32})
    b = NxEigen.tensor([[2.0, 4.0]], type: {:f, 32})

    assert Nx.to_binary(Nx.equal(a, b)) == Nx.to_binary(Nx.tensor([[0, 0]], type: {:u, 8}))
    assert Nx.to_binary(Nx.greater(a, b)) == Nx.to_binary(Nx.tensor([[0, 1]], type: {:u, 8}))
  end

  test "upcasting" do
    a = NxEigen.tensor([[1.0, 2.0]], type: {:f, 32})
    b = NxEigen.tensor([[3.0, 4.0]], type: {:f, 64})

    # Addition with different types
    res = Nx.add(a, b)
    assert res.type == {:f, 64}
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[4.0, 6.0]], type: {:f, 64}))

    # Comparison with different types
    res = Nx.greater(a, b)
    assert res.type == {:u, 8}
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[0, 0]], type: {:u, 8}))
  end

  test "as_type" do
    t = NxEigen.tensor([[1.0, 2.0]], type: {:f, 32})
    t64 = Nx.as_type(t, {:f, 64})
    assert t64.type == {:f, 64}
    assert Nx.to_binary(t64) == Nx.to_binary(Nx.tensor([[1.0, 2.0]], type: {:f, 64}))

    u8 = Nx.as_type(t, {:u, 8})
    assert u8.type == {:u, 8}
    assert Nx.to_binary(u8) == Nx.to_binary(Nx.tensor([[1, 2]], type: {:u, 8}))
  end

  test "all integer types" do
    types = [
      {:u, 8}, {:u, 16}, {:u, 32}, {:u, 64},
      {:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}
    ]

    for type <- types do
      t = NxEigen.tensor([[1, 2], [3, 4]], type: type)
      assert t.type == type
      assert Nx.to_binary(t) == Nx.to_binary(Nx.tensor([[1, 2], [3, 4]], type: type))

      res = Nx.add(t, t)
      assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[2, 4], [6, 8]], type: type))
    end
  end

  test "complex types" do
    t = NxEigen.tensor([[1.0, 2.0]], type: {:c, 64})
    assert t.type == {:c, 64}

    # We can't easily compare complex binary because of endianness/format,
    # but we can check if operations work
    res = Nx.add(t, t)
    # Nx.to_binary(res) should match standard Nx complex binary
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[2.0, 4.0]], type: {:c, 64}))
  end

  test "sum reduction" do
    t = NxEigen.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: {:f, 32})

    # Sum all elements (axes: [0, 1])
    res = Nx.sum(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(21.0, type: {:f, 32}))

    # Sum along axis 0
    res = Nx.sum(t, axes: [0])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([5.0, 7.0, 9.0], type: {:f, 32}))

    # Sum along axis 1
    res = Nx.sum(t, axes: [1])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([6.0, 15.0], type: {:f, 32}))
  end

  test "product reduction" do
    t = NxEigen.tensor([[2.0, 3.0], [4.0, 5.0]], type: {:f, 32})

    # Product of all elements
    res = Nx.product(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(120.0, type: {:f, 32}))

    # Product along axis 0
    res = Nx.product(t, axes: [0])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([8.0, 15.0], type: {:f, 32}))
  end

  test "reduce_max and reduce_min" do
    t = NxEigen.tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 7.0]], type: {:f, 32})

    # Max of all elements
    res = Nx.reduce_max(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(9.0, type: {:f, 32}))

    # Max along axis 0
    res = Nx.reduce_max(t, axes: [0])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([9.0, 5.0, 7.0], type: {:f, 32}))

    # Min of all elements
    res = Nx.reduce_min(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(1.0, type: {:f, 32}))

    # Min along axis 1
    res = Nx.reduce_min(t, axes: [1])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([1.0, 2.0], type: {:f, 32}))
  end

  test "all and any reductions" do
    t = NxEigen.tensor([[1, 0], [1, 1]], type: {:u, 8})

    # All (logical AND)
    res = Nx.all(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(0, type: {:u, 8}))

    res = Nx.all(t, axes: [0])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([1, 0], type: {:u, 8}))

    # Any (logical OR)
    res = Nx.any(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(1, type: {:u, 8}))

    res = Nx.any(t, axes: [1])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([1, 1], type: {:u, 8}))
  end

  # TODO: Fix argmax/argmin multi-column case
  # test "argmax and argmin" do
  #   t = NxEigen.tensor([[1.0, 5.0, 3.0], [9.0, 2.0, 7.0]], type: {:f, 32})

  #   # Argmax - flatten (defaults to s32)
  #   res = Nx.argmax(t)
  #   assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(3, type: {:s, 32}))

  #   # Argmax along axis 0
  #   res = Nx.argmax(t, axis: 0)
  #   assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([1, 0, 1], type: {:s, 32}))

  #   # Argmax along axis 1
  #   res = Nx.argmax(t, axis: 1)
  #   assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([1, 0], type: {:s, 32}))

  #   # Argmin - flatten
  #   res = Nx.argmin(t)
  #   assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(0, type: {:s, 32}))

  #   # Argmin along axis 1
  #   res = Nx.argmin(t, axis: 1)
  #   assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([0, 1], type: {:s, 32}))
  # end

  test "reductions with integer types" do
    t = NxEigen.tensor([[1, 2], [3, 4]], type: {:s, 32})

    res = Nx.sum(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(10, type: {:s, 32}))

    res = Nx.product(t)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor(24, type: {:s, 32}))
  end

  test "slice operation" do
    t = NxEigen.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: {:f, 32})

    # Slice first row
    res = Nx.slice(t, [0, 0], [1, 3])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[1.0, 2.0, 3.0]], type: {:f, 32}))

    # Slice first column
    res = Nx.slice(t, [0, 0], [2, 1])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[1.0], [4.0]], type: {:f, 32}))

    # Slice with stride [1, 2] - slices have output shape determined by lengths,
    # not stride, so [2, 2] with stride [1, 2] gives {2, 1} actual shape
    res = Nx.slice(t, [0, 0], [2, 1], strides: [1, 2])
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[1.0], [4.0]], type: {:f, 32}))
  end

  test "put_slice operation" do
    t = NxEigen.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: {:f, 32})
    slice_val = NxEigen.tensor([[10.0, 20.0]], type: {:f, 32})

    res = Nx.put_slice(t, [0, 1], slice_val)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[1.0, 10.0, 20.0], [4.0, 5.0, 6.0]], type: {:f, 32}))
  end

  test "select operation" do
    pred = NxEigen.tensor([[1, 0], [0, 1]], type: {:u, 8})
    on_true = NxEigen.tensor([[10.0, 20.0], [30.0, 40.0]], type: {:f, 32})
    on_false = NxEigen.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})

    res = Nx.select(pred, on_true, on_false)
    assert Nx.to_binary(res) == Nx.to_binary(Nx.tensor([[10.0, 2.0], [3.0, 40.0]], type: {:f, 32}))
  end

  # Linear Algebra
  test "dot - 2D matrix multiplication" do
    a = NxEigen.tensor([[1, 2], [3, 4]])
    b = NxEigen.tensor([[5, 6], [7, 8]])
    result = Nx.dot(a, b)
    # [1*5+2*7, 1*6+2*8] = [19, 22]
    # [3*5+4*7, 3*6+4*8] = [43, 50]
    assert Nx.to_list(result) == [[19, 22], [43, 50]]
  end

  test "dot - vector dot product" do
    a = NxEigen.tensor([1, 2, 3])
    b = NxEigen.tensor([4, 5, 6])
    result = Nx.dot(a, b)
    # 1*4 + 2*5 + 3*6 = 32
    assert Nx.to_number(result) == 32
  end
end
