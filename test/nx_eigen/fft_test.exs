defmodule NxEigen.FFTTest do
  use ExUnit.Case, async: true

  # Nx's own FFT doctests only use handfuls of samples, which leaves the
  # interesting paths of the FFT backends untested: kissfft's radix-2/3/4/5
  # butterflies, and the Bluestein convolution the `eigen` backend falls back to
  # once the largest prime factor exceeds 64. These lengths straddle that
  # threshold, and the binary backend is the oracle.
  @radix_lengths [2, 7, 13, 37, 64, 128, 1000]
  @bluestein_lengths [101, 127, 1021]

  @tolerances %{f32: 1.0e-4, f64: 1.0e-10}

  defp signal(n), do: for(i <- 0..(n - 1), do: :math.sin(i * 0.7) + 0.3 * i)

  defp relative_error(actual, expected) do
    error =
      actual
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.subtract(expected)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    scale = expected |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    error / max(scale, 1.0e-12)
  end

  for op <- [:fft, :ifft],
      length <- @radix_lengths ++ @bluestein_lengths,
      type <- [:f32, :f64] do
    test "#{op} of #{length} #{type} samples matches the binary backend" do
      data = signal(unquote(length))

      expected =
        data
        |> Nx.tensor(type: unquote(type), backend: Nx.BinaryBackend)
        |> then(&apply(Nx, unquote(op), [&1]))

      actual =
        data
        |> Nx.tensor(type: unquote(type), backend: NxEigen.Backend)
        |> then(&apply(Nx, unquote(op), [&1]))

      assert relative_error(actual, expected) < @tolerances[unquote(type)]
    end
  end

  for length <- @radix_lengths ++ @bluestein_lengths do
    test "ifft of fft returns the original #{length} samples" do
      tensor = Nx.tensor(signal(unquote(length)), type: :f64, backend: NxEigen.Backend)

      round_tripped = tensor |> Nx.fft() |> Nx.ifft()

      assert relative_error(round_tripped, Nx.backend_transfer(tensor, Nx.BinaryBackend)) <
               1.0e-10
    end
  end

  test "a length with a large prime factor completes promptly" do
    # Without Bluestein this is a 9-second O(n²) transform on the `eigen`
    # backend, so a generous ceiling still catches a regression.
    tensor = Nx.tensor(signal(65_521), type: :f64, backend: NxEigen.Backend)

    {microseconds, _} = :timer.tc(fn -> Nx.fft(tensor) end)

    assert microseconds < 1_000_000
  end
end
