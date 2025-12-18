defmodule NxEigen.NIF do
  @on_load :load_nif

  def load_nif do
    path = :code.priv_dir(:nx_eigen) |> Path.join("libnx_eigen")
    :erlang.load_nif(to_charlist(path), 0)
  end

  # Native functions
  def from_binary(binary, type, shape) do
    # Fine can decode {atom, integer} tuples directly into our C++ enum
    # but the shape needs to be a list for std::vector<int64_t> decoding
    from_binary_nif(binary, type, Tuple.to_list(shape))
  end

  defp from_binary_nif(_binary, _type, _shape), do: :erlang.nif_error(:nif_not_loaded)

  def to_binary(_resource), do: :erlang.nif_error(:nif_not_loaded)
  def add(_left, _right), do: :erlang.nif_error(:nif_not_loaded)
  def subtract(_left, _right), do: :erlang.nif_error(:nif_not_loaded)
  def multiply(_left, _right), do: :erlang.nif_error(:nif_not_loaded)
end
