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
end
