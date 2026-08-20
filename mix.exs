Code.require_file("precompiler.exs", __DIR__)

defmodule NxEigen.MixProject do
  use Mix.Project

  @version "0.1.1"
  @source_url "https://github.com/polvalente/nx_eigen"
  @cc_template "<%= cc %>"
  @cxx_template "<%= cxx %>"
  @uno_q_linux_flags "-march=armv8-a+crypto+crc -mtune=cortex-a53 -mfix-cortex-a53-835769 -mfix-cortex-a53-843419"

  # Mirrors TARGET_GCC_FLAGS from the Nerves systems built on a Cortex-A7 (such
  # as nerves_system_trellis), minus -fPIE/-pie, which conflict with building a
  # shared library. Symbols are stripped because Nerves strips target binaries
  # by default and the debug info is a third of the size of the NIF.
  @cortex_a7_flags "-mabi=aapcs-linux -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -marm -fstack-protector-strong -s -Wl,-z,now -Wl,-z,relro"
  @cortex_a7_target "armv7-cortex-a7-linux-gnueabihf"
  @nerves_armv7_prefix "armv7-nerves-linux-gnueabihf-"

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
      # Force source build in dev/test so only_listed_targets: true doesn't cause
      # cc_precompiler to return :ignore and silently skip NIF compilation.
      make_force_build: Mix.env() != :prod,
      make_precompiler: {:nif, NxEigen.Precompiler},
      make_precompiler_url: "#{@source_url}/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "libnx_eigen",
      make_precompiler_priv_paths: ["libnx_eigen.so"],
      make_precompiler_nif_versions: [versions: ["2.17"]],
      cc_precompiler: [
        cleanup: "clean",
        cmake_lists_path: "CMakeLists.txt",
        cmake_build_type: "Release",
        cmake_flags: ["-DNX_EIGEN_FFT_LIB=fftw"],
        # Restrict targets to those we explicitly list for the current OS.
        # This prevents macOS runners from attempting to build both archs in one job.
        only_listed_targets: true,
        compilers: %{
          {:unix, :linux} => linux_targets_at_compile_time(),
          {:unix, :darwin} => macos_targets()
        }
      ],

      # Package configuration
      package: package(),
      description: "High-performance numerical computing with Eigen backend",

      # Documentation configuration
      docs: docs(),
      name: "NxEigen",
      source_url: @source_url
    ]
  end

  defp make_env do
    fine_include =
      if Code.ensure_loaded?(Fine) do
        Fine.include_dir()
      else
        Path.join([File.cwd!(), "deps", "fine", "c_include"])
      end

    %{"FINE_INCLUDE" => fine_include}
    |> Map.merge(target_fft_lib())
    |> forward_env("NX_EIGEN_FFT_LIB")
    |> forward_env("NX_EIGEN_FFT_SO")
  end

  # Nerves systems don't ship FFTW, so the Cortex-A7 target builds against
  # Eigen's own FFT module instead. `forward_env` runs after this, so an
  # explicit NX_EIGEN_FFT_LIB still wins.
  defp target_fft_lib do
    case System.get_env("PRECOMPILE_TARGET") do
      @cortex_a7_target -> %{"NX_EIGEN_FFT_LIB" => "eigen"}
      _ -> %{}
    end
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
      {:cc_precompiler, "~> 0.1", runtime: false},
      {:fine, "~> 0.1.0"},
      {:ex_doc, "~>  0.40", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      files: [
        "lib",
        "c_src",
        "Makefile",
        "CMakeLists.txt",
        "checksum.exs",
        "mix.exs",
        "precompiler.exs",
        "README.md",
        "LICENSE"
      ],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "README",
      source_ref: "v#{@version}",
      source_url: @source_url,
      extras: [
        "README.md",
        "PRECOMPILATION.md"
      ],
      groups_for_modules: [
        Core: [NxEigen, NxEigen.Backend]
      ]
    ]
  end

  defp toolchain(c_compiler, cxx_compiler, cc_cmd, cxx_cmd) do
    {c_compiler, cxx_compiler, cc_cmd, cxx_cmd}
  end

  defp gcc_toolchain(cc_cmd, cxx_cmd) do
    toolchain("gcc", "g++", cc_cmd, cxx_cmd)
  end

  defp clang_toolchain(cc_cmd, cxx_cmd) do
    toolchain("clang", "clang++", cc_cmd, cxx_cmd)
  end

  defp linux_target(name, cc_cmd \\ @cc_template, cxx_cmd \\ @cxx_template) do
    %{name => gcc_toolchain(cc_cmd, cxx_cmd)}
  end

  defp macos_target(name, arch) do
    %{name => clang_toolchain("#{@cc_template} -arch #{arch}", "#{@cxx_template} -arch #{arch}")}
  end

  # On Linux, determine targets at compile time (not at project definition time)
  # This avoids Mix caching issues
  defp linux_targets_at_compile_time do
    # Check if specific target is requested (e.g., from Docker)
    target = System.get_env("PRECOMPILE_TARGET")

    case target do
      nil ->
        # Build for current architecture only
        native_linux_target()

      "all" ->
        # Used for generating checksums across *all* published Linux targets
        all_linux_targets()

      "aarch64-arduino-uno-q-linux-gnu" ->
        # Arduino Uno Q optimized target (ARM64 with specific flags)
        # Return ONLY this target, not the base aarch64-linux-gnu
        linux_target(
          "aarch64-arduino-uno-q-linux-gnu",
          "#{@cc_template} #{@uno_q_linux_flags}",
          "#{@cxx_template} #{@uno_q_linux_flags}"
        )

      target ->
        # Build for specific target
        case target do
          "x86_64-linux-gnu" ->
            linux_target("x86_64-linux-gnu")

          "aarch64-linux-gnu" ->
            linux_target("aarch64-linux-gnu")

          @cortex_a7_target ->
            cortex_a7_target()

          "riscv64-linux-gnu" ->
            linux_target("riscv64-linux-gnu")

          _ ->
            IO.warn("Unknown PRECOMPILE_TARGET: #{target}, falling back to native")
            native_linux_target()
        end
    end
  end

  defp all_linux_targets do
    linux_target("x86_64-linux-gnu")
    |> Map.merge(linux_target("aarch64-linux-gnu"))
    |> Map.merge(
      linux_target(
        "aarch64-arduino-uno-q-linux-gnu",
        "#{@cc_template} #{@uno_q_linux_flags}",
        "#{@cxx_template} #{@uno_q_linux_flags}"
      )
    )
    |> Map.merge(cortex_a7_target())
  end

  # Built with the Nerves toolchain so that the glibc and libstdc++ the NIF links
  # against match the ones in the Nerves system.
  defp cortex_a7_target do
    %{
      @cortex_a7_target =>
        toolchain(
          "#{@nerves_armv7_prefix}gcc",
          "#{@nerves_armv7_prefix}g++",
          "#{@cc_template} #{@cortex_a7_flags}",
          "#{@cxx_template} #{@cortex_a7_flags}"
        )
    }
  end

  # Native Linux target based on current architecture
  defp native_linux_target do
    case :erlang.system_info(:system_architecture) |> to_string() do
      "x86_64" <> _ ->
        linux_target("x86_64-linux-gnu")

      "aarch64" <> _ ->
        linux_target("aarch64-linux-gnu")

      arch ->
        # Fallback: try native compilation
        IO.warn("Unknown Linux architecture: #{arch}, attempting native compilation")

        linux_target("#{arch}-linux-gnu")
    end
  end

  # On macOS, only build for the current architecture since FFTW is dynamically linked
  # and Homebrew provides architecture-specific binaries
  defp macos_targets do
    case System.get_env("PRECOMPILE_TARGET") do
      "all" ->
        macos_target("x86_64-apple-darwin", "x86_64")
        |> Map.merge(macos_target("aarch64-apple-darwin", "arm64"))

      _ ->
        macos_targets_native()
    end
  end

  defp macos_targets_native do
    case :erlang.system_info(:system_architecture) |> to_string() do
      "aarch64" <> _ ->
        macos_target("aarch64-apple-darwin", "arm64")

      "arm64" <> _ ->
        macos_target("aarch64-apple-darwin", "arm64")

      "x86_64" <> _ ->
        macos_target("x86_64-apple-darwin", "x86_64")

      arch ->
        # Fallback: build both if we can't detect
        IO.warn("Unknown macOS architecture: #{arch}, building for both x86_64 and arm64")

        macos_target("x86_64-apple-darwin", "x86_64")
        |> Map.merge(macos_target("aarch64-apple-darwin", "arm64"))
    end
  end
end
