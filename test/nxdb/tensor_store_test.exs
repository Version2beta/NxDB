defmodule NxDB.TensorStoreTest do
  use ExUnit.Case, async: true
  doctest NxDB.TensorStore

  setup_all do
    if function_exported?(NxDB.TensorStore, :start_link, 1) do
      {:ok, pid} = NxDB.TensorStore.start_link(storage_path: "test_storage.dat")

      on_exit(fn ->
        if Process.alive?(pid), do: GenServer.stop(pid)
      end)

      {:ok, pid: pid}
    else
      :ok
    end
  end
end
