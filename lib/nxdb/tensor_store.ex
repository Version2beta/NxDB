defmodule NxDB.TensorStore do
  @moduledoc """
  The `NxDB.TensorStore` module provides a memory-mapped storage layer for Nx tensors,
  enabling high-performance and scalable persistence without the overhead of full serialization.

  ## Key Features

    - **Memory-Mapped Storage:**
      Tensors are stored directly on disk in a binary format and mapped into the process's address space,
      leveraging the operating system’s paging mechanism for on-demand data loading and reduced memory overhead.

    - **Native Tensor Format:**
      Tensors are stored in their native format, eliminating the need for costly serialization/deserialization,
      which lowers latency during tensor operations.

    - **Batch Processing and Streaming:**
      Supports both individual and batch operations, making it efficient to handle large-scale inserts and retrievals.

    - **Integration with Metadata and ANN Components:**
      Seamlessly works with the ETS-based metadata index for quick tensor ID lookups and the HNSW-based ANN index
      for fast approximate nearest neighbor searches on 1D tensors.

    - **Scalability and Fault Tolerance:**
      The memory-mapped approach supports datasets larger than system memory and enhances persistence,
      ensuring robust recovery in case of failures.

  ## Public API

  The module exposes the following functions:

    - `start_link/1` - Starts the tensor store process and performs necessary initialization.
    - `insert/2` - Inserts a single tensor.
    - `get/1` - Retrieves a single tensor by its ID.
    - `batch_insert/1` - Inserts multiple tensors in one operation.
    - `batch_get/1` - Retrieves multiple tensors by their IDs.
    - `stream/0` - Streams all tensors for lazy processing.
  """

  @doc """
  Starts the TensorStore process.

  Accepts a keyword list of options (e.g., the storage file path) and returns `{:ok, _pid}`.

  ## Example

      iex> {:ok, _pid} = NxDB.TensorStore.start_link(storage_path: "tmp/storage.dat")
  """
  def start_link(opts) do
    NxDB.TensorStore.Server.start_link(opts)
  end

  @doc """
  Inserts a new tensor into the store.

  Returns `:ok` if the message is successfully dispatched.

  ## Examples

      iex> {:ok, _pid} = NxDB.TensorStore.start_link(storage_path: "tmp/storage.dat")
      iex> tensor = Nx.tensor([1, 2, 3])
      iex> :ok = NxDB.TensorStore.insert("tensor_1", tensor)
  """
  def insert(tensor_id, nx_tensor) do
    GenServer.cast(NxDB.TensorStore.Server, {:insert, tensor_id, nx_tensor})
  end

  @doc """
  Retrieves a tensor from the store by its ID.

  Returns `{:ok, tensor}` if found, or `{:error, :not_found}` if the tensor does not exist.

  ## Examples

      iex> {:ok, pid} = NxDB.TensorStore.start_link(storage_path: "tmp/storage.dat")
      iex> tensor = Nx.tensor([1, 2, 3])
      iex> :ok = NxDB.TensorStore.insert("tensor_1", tensor)
      iex> {:ok, tensor} = NxDB.TensorStore.get("tensor_1")
      iex> GenServer.stop(pid)
      iex> Nx.shape(tensor)
      {3}
  """
  def get(tensor_id) do
    GenServer.call(NxDB.TensorStore.Server, {:get, tensor_id})
  end

  @doc """
  Inserts multiple tensors in a batch.

  Accepts a list of tuples in the form `[{tensor_id, nx_tensor}, ...]` and returns `:ok`
  if the message is successfully dispatched.

  ## Examples

      iex> {:ok, _pid} = NxDB.TensorStore.start_link(storage_path: "tmp/storage.dat")
      iex> tensor1 = Nx.tensor([1, 2, 3])
      iex> tensor2 = Nx.tensor([4, 5, 6])
      iex> :ok = NxDB.TensorStore.batch_insert([{"tensor_1", tensor1}, {"tensor_2", tensor2}])
  """
  def batch_insert(tensor_list) when is_list(tensor_list) do
    GenServer.cast(NxDB.TensorStore.Server, {:batch_insert, tensor_list})
  end

  @doc """
  Retrieves multiple tensors from the store.

  Accepts a list of tensor IDs and returns a map with tensor IDs as keys and their corresponding tensors as values.

  ## Examples

      iex> {:ok, _pid} = NxDB.TensorStore.start_link(storage_path: "tmp/storage.dat")
      iex> NxDB.TensorStore.batch_insert([
      ...>   {"tensor_1", Nx.tensor([1, 2, 3])},
      ...>   {"tensor_2", Nx.tensor([4, 5, 6])}
      ...> ])
      :ok
      iex> tensors = NxDB.TensorStore.batch_get(["tensor_1", "tensor_2"])
      iex> Nx.shape(tensors["tensor_1"])
      {3}
      iex> Nx.shape(tensors["tensor_2"])
      {3}
      iex> Nx.to_flat_list(tensors["tensor_1"])
      [1, 2, 3]
      iex> Nx.to_flat_list(tensors["tensor_2"])
      [4, 5, 6]
  """
  def batch_get(tensor_ids) when is_list(tensor_ids) do
    GenServer.call(NxDB.TensorStore.Server, {:batch_get, tensor_ids})
  end

  @doc """
  Streams tensors from the store.

  Returns a stream of `{tensor_id, tensor}` tuples for lazy processing of stored tensors.

  ## Examples

      iex> {:ok, _pid} = NxDB.TensorStore.start_link(storage_path: "tmp/storage.dat")
      iex> NxDB.TensorStore.batch_insert([
      ...>   {"tensor_1", Nx.tensor([1, 2, 3])},
      ...>   {"tensor_2", Nx.tensor([4, 5, 6])}
      ...> ])
      :ok
      iex> stream = NxDB.TensorStore.stream()
      iex> result = Enum.take(stream, 2)
      iex> [{"tensor_1", tensor1}, {"tensor_2", tensor2}] = result
      iex> Nx.shape(tensor1)
      {3}
      iex> Nx.shape(tensor2)
      {3}
      iex> Nx.to_flat_list(tensor1)
      [1, 2, 3]
      iex> Nx.to_flat_list(tensor2)
      [4, 5, 6]
  """
  def stream do
    GenServer.call(NxDB.TensorStore.Server, :stream)
    |> Stream.map(fn {k, v} -> {k, v} end)
  end
end
