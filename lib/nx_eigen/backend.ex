defmodule NxEigen.Backend do
  @behaviour Nx.Backend

  defstruct [:state, :id]

  @impl true
  def init(opts) do
    %__MODULE__{id: opts[:id] || make_ref()}
  end

  @impl true
  def from_binary(tensor, binary, _backend_opts) do
    state = NxEigen.NIF.from_binary(binary, tensor.type, tensor.shape)
    %{tensor | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{state: state}}, _limit) do
    NxEigen.NIF.to_binary(state)
  end

  @impl true
  def inspect(%Nx.Tensor{} = tensor, inspect_opts) do
    Nx.Backend.inspect(tensor, inspect_opts)
  end

  @impl true
  def add(out, l, r) do
    state = NxEigen.NIF.add(l.data.state, r.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def subtract(out, l, r) do
    state = NxEigen.NIF.subtract(l.data.state, r.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end

  @impl true
  def multiply(out, l, r) do
    state = NxEigen.NIF.multiply(l.data.state, r.data.state)
    %{out | data: %__MODULE__{state: state, id: make_ref()}}
  end
end
