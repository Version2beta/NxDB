defmodule NxDB.TensorStore.Server do
  use GenServer

  @moduledoc false
  # Internal GenServer implementation for the NxDB.TensorStore.
  # This module manages the state and handles message passing for tensor operations.

  @doc false
  def start_link(opts) do
    case GenServer.start_link(__MODULE__, opts, name: __MODULE__) do
      {:ok, pid} -> {:ok, pid}
      # Prevents failures when already started
      {:error, {:already_started, pid}} -> {:ok, pid}
    end
  end

  @impl true
  def init(opts) do
    # Simulate initialization by setting up an in-memory store.
    # In a full implementation, this would open the file and set up memory mapping.
    state = %{
      storage_path: opts[:storage_path],
      store: %{}
    }

    {:ok, state}
  end

  @impl true
  def handle_cast({:insert, tensor_id, nx_tensor}, state) do
    new_store = Map.put(state.store, tensor_id, nx_tensor)
    {:noreply, %{state | store: new_store}}
  end

  @impl true
  def handle_cast({:batch_insert, tensor_list}, state) do
    new_store =
      Enum.reduce(tensor_list, state.store, fn {tensor_id, nx_tensor}, acc ->
        Map.put(acc, tensor_id, nx_tensor)
      end)

    {:noreply, %{state | store: new_store}}
  end

  @impl true
  def handle_call({:get, tensor_id}, _from, state) do
    case Map.fetch(state.store, tensor_id) do
      {:ok, tensor} ->
        {:reply, {:ok, tensor}, state}

      :error ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:batch_get, tensor_ids}, _from, state) do
    result =
      Enum.reduce(tensor_ids, %{}, fn id, acc ->
        case Map.fetch(state.store, id) do
          {:ok, tensor} -> Map.put(acc, id, tensor)
          :error -> acc
        end
      end)

    {:reply, result, state}
  end

  @impl true
  def handle_call(:stream, _from, state) do
    {:reply, state.store, state}
  end
end
