defmodule NxEigen do
  @moduledoc """
  Documentation for `NxEigen`.
  """

  @doc """
  Returns a new tensor with the given data using the NxEigen backend.
  """
  def tensor(data, opts \\ []) do
    Nx.tensor(data, Keyword.put(opts, :backend, NxEigen.Backend))
  end
end
