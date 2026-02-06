defmodule NxEigen.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_eigen,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      make_env: make_env(),
      deps: deps()
    ]
  end

  defp make_env do
    %{}
    |> forward_env("NX_EIGEN_FFT_LIB")
    |> forward_env("NX_EIGEN_FFT_SO")
  end

  defp forward_env(env, var) do
    case System.get_env(var) do
      nil -> env
      val -> Map.put(env, var, val)
    end
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.10"},
      {:elixir_make, "~> 0.8", runtime: false},
      {:fine, "~> 0.1.0"}
    ]
  end
end
