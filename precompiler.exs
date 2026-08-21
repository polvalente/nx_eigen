defmodule NxEigen.Precompiler do
  @moduledoc false

  # Every Nerves system exports TARGET_ARCH=arm, TARGET_OS=linux and
  # TARGET_ABI=gnueabihf for 32-bit ARM boards, so cc_precompiler resolves ARMv6
  # (arm1176jzf_s) and ARMv7 (cortex_a7, cortex_a53) devices to the same
  # `arm-linux-gnueabihf` triplet even though a binary built for one will not run
  # on the others. TARGET_CPU is the only thing that tells them apart, so the
  # triplet is refined with it here.
  #
  # The `:fetch` targets are listed rather than delegated because cc_precompiler
  # derives them from the compilers it can find on the machine doing the
  # fetching, which never includes cross-compiled targets.
  #
  # Calls into cc_precompiler go through apply/3 because this file is evaluated
  # every time mix.exs is loaded, including before the dependency is compiled.

  @published_targets [
    "aarch64-apple-darwin",
    "aarch64-arduino-uno-q-linux-gnu",
    "aarch64-linux-gnu",
    "armv7-cortex-a7-linux-gnueabihf",
    "x86_64-apple-darwin",
    "x86_64-linux-gnu"
  ]

  def all_supported_targets(:fetch), do: @published_targets
  def all_supported_targets(:compile), do: cc_precompiler(:all_supported_targets, [:compile])

  def build_native(args), do: cc_precompiler(:build_native, [args])

  def current_target do
    case cc_precompiler(:current_target, []) do
      {:ok, "arm-linux-gnueabihf"} -> {:ok, arm_target(System.get_env("TARGET_CPU"))}
      other -> other
    end
  end

  def post_precompile_target(target), do: cc_precompiler(:post_precompile_target, [target])

  def precompile(args, target), do: cc_precompiler(:precompile, [args, target])

  def unavailable_target(target), do: cc_precompiler(:unavailable_target, [target])

  defp arm_target("cortex_a7"), do: "armv7-cortex-a7-linux-gnueabihf"
  defp arm_target("arm1176" <> _), do: "armv6-linux-gnueabihf"
  defp arm_target(_), do: "arm-linux-gnueabihf"

  defp cc_precompiler(function, args), do: apply(CCPrecompiler, function, args)
end
