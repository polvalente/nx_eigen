defmodule NxEigen.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/YOUR_USERNAME/nx_eigen"  # TODO: Update with your GitHub repo URL

  def project do
    [
      app: :nx_eigen,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_targets: ["all"],
      make_clean: ["clean"],
      make_env: make_env(),
      deps: deps(),

      # Precompilation configuration
      make_precompiler: {:nif, CCPrecompiler},
      make_precompiler_url: "#{@source_url}/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "libnx_eigen",
      make_precompiler_priv_paths: ["libnx_eigen.so"],
      make_precompiler_nif_versions: [versions: ["2.15", "2.16", "2.17"]],
      cc_precompiler: [
        cleanup: "clean"
      ],
      cc_precompile: cc_precompile(),

      # Package configuration
      package: package(),
      description: "High-performance numerical computing with Eigen backend"
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
      {:cc_precompiler, "~> 0.1.0", runtime: false, github: "cocoa-xu/cc_precompiler"},
      {:fine, "~> 0.1.0"}
    ]
  end

  defp package do
    [
      files: [
        "lib",
        "c_src",
        "Makefile",
        "CMakeLists.txt",
        "checksum-nx_eigen.exs",
        "mix.exs",
        "README.md",
        "LICENSE"
      ],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp cc_precompile do
    [
      compilers: %{
        # Linux targets - only build for current architecture
        {:unix, :linux} => linux_targets(),
        # macOS targets - only build for current architecture
        {:unix, :darwin} => macos_targets()
      }
    ]
  end

  # On Linux, build for the current architecture using native compilation
  # Using gcc/g++ without cross-compilation prefix for native builds
  # For local cross-compilation, use Docker (see scripts/precompile-docker.sh)
  defp linux_targets do
    case :erlang.system_info(:system_architecture) |> to_string() do
      "x86_64" <> _ ->
        %{
          "x86_64-linux-gnu" => {
            "gcc",
            "g++",
            "<%= cc %>",
            "<%= cxx %>"
          }
        }

      "aarch64" <> _ ->
        %{
          "aarch64-linux-gnu" => {
            "gcc",
            "g++",
            "<%= cc %>",
            "<%= cxx %>"
          }
        }

      arch ->
        # Fallback: try native compilation
        IO.warn("Unknown Linux architecture: #{arch}, attempting native compilation")

        %{
          "#{arch}-linux-gnu" => {
            "gcc",
            "g++",
            "<%= cc %>",
            "<%= cxx %>"
          }
        }
    end
  end

  # On macOS, only build for the current architecture since FFTW is dynamically linked
  # and Homebrew provides architecture-specific binaries
  defp macos_targets do
    case :erlang.system_info(:system_architecture) |> to_string() do
      "aarch64" <> _ ->
        %{
          "aarch64-apple-darwin" => {
            "clang",
            "clang++",
            "<%= cc %> -arch arm64",
            "<%= cxx %> -arch arm64"
          }
        }

      "x86_64" <> _ ->
        %{
          "x86_64-apple-darwin" => {
            "clang",
            "clang++",
            "<%= cc %> -arch x86_64",
            "<%= cxx %> -arch x86_64"
          }
        }

      arch ->
        # Fallback: build both if we can't detect
        IO.warn("Unknown macOS architecture: #{arch}, building for both x86_64 and arm64")

        %{
          "x86_64-apple-darwin" => {
            "clang",
            "clang++",
            "<%= cc %> -arch x86_64",
            "<%= cxx %> -arch x86_64"
          },
          "aarch64-apple-darwin" => {
            "clang",
            "clang++",
            "<%= cc %> -arch arm64",
            "<%= cxx %> -arch arm64"
          }
        }
    end
  end
end
