defmodule NxEigen.PrecompilerTest do
  use ExUnit.Case, async: false

  @env_keys ~w(TARGET_ARCH TARGET_OS TARGET_ABI TARGET_CPU NERVES_SDK_SYSROOT)

  setup do
    previous = Map.new(@env_keys, &{&1, System.get_env(&1)})

    on_exit(fn ->
      Enum.each(previous, fn
        {key, nil} -> System.delete_env(key)
        {key, value} -> System.put_env(key, value)
      end)
    end)

    :ok
  end

  defp with_target(env, fun) do
    Enum.each(@env_keys, &System.delete_env/1)
    Enum.each(env, fn {key, value} -> System.put_env(key, value) end)
    fun.()
  end

  test "aarch64 firmware build selects the nerves target" do
    with_target(
      %{
        "TARGET_ARCH" => "aarch64",
        "TARGET_OS" => "linux",
        "TARGET_ABI" => "gnu",
        "NERVES_SDK_SYSROOT" => "/tmp"
      },
      fn ->
        assert NxEigen.Precompiler.current_target() == {:ok, "aarch64-nerves-linux-gnu"}
      end
    )
  end

  test "desktop aarch64 keeps the generic target" do
    with_target(
      %{"TARGET_ARCH" => "aarch64", "TARGET_OS" => "linux", "TARGET_ABI" => "gnu"},
      fn ->
        assert NxEigen.Precompiler.current_target() == {:ok, "aarch64-linux-gnu"}
      end
    )
  end

  test "arduino uno q keeps its target when a sysroot is set" do
    with_target(
      %{
        "TARGET_ARCH" => "aarch64",
        "TARGET_OS" => "arduino-uno-q-linux",
        "TARGET_ABI" => "gnu",
        "NERVES_SDK_SYSROOT" => "/tmp"
      },
      fn ->
        assert NxEigen.Precompiler.current_target() == {:ok, "aarch64-arduino-uno-q-linux-gnu"}
      end
    )
  end

  test "cortex-a7 firmware build keeps the armv7 target" do
    with_target(
      %{
        "TARGET_ARCH" => "arm",
        "TARGET_OS" => "linux",
        "TARGET_ABI" => "gnueabihf",
        "TARGET_CPU" => "cortex_a7",
        "NERVES_SDK_SYSROOT" => "/tmp"
      },
      fn ->
        assert NxEigen.Precompiler.current_target() == {:ok, "armv7-cortex-a7-linux-gnueabihf"}
      end
    )
  end
end
