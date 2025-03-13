# NxDB.TensorStore.FileManager

## Overview

`NxDB.TensorStore.FileManager` is responsible for **managing the storage and retrieval of tensors** within NxDB. It provides efficient methods to store, retrieve, and manage both **1D indexed vectors** for ANN search and **full ND tensors**, which are stored separately in binary files.

### **Key Features**

- **Optimized Cluster-Based Storage**
  - Stores **1D vectors within IVF clusters** for ANN search.
  - **HNSW graph management** at both **global (cluster) and local (vector) levels**.
- **LRU-Based Cluster Caching**
  - Frequently accessed clusters are kept in memory **to reduce disk I/O**.
- **GPU-Optimized Processing**
  - Uses `Nx.Defn` for **batch processing and ANN acceleration**.
- **Binary File Storage for ND Tensors**
  - ND tensors are stored **as individual binary files, retrievable by ID**.
  - Uses **metadata indexing** for efficient lookup and filtering.
- **1D Vector Compression & ANN Linking**
  - Users can **optionally store compressed 1D representations** for ANN search.
  - **User determines** optimal compression strategy.
  - Links compressed vectors to full ND tensors using **metadata references**.
- **Explicit Memory Management API**
  - Uses `sync`, `sync_lock`, and `sync_unlock` for **controlled side effects**.
  - Guarantees **pure functions operate only on in-memory data**.

---

## **Storage Strategy**

### **1️⃣ 1D Vector Storage for ANN Search**

- **Users generate raw or compressed 1D vectors** and store them separately.
- **IVF clusters organize these vectors for ANN search**.
- **Local HNSW graphs connect vectors within a cluster**.
- **A global HNSW graph connects clusters for faster retrieval**.

### **2️⃣ Metadata Indexing**

- **Stores shape, type, and tensor references, plus user specified metadata**.
- **Used for fast lookup and filtering of ND tensors**.
- **Links ANN search results to full ND tensors** when provided by the user.

### **3️⃣ ND Tensor Storage**

- **Stored separately from ANN index** in binary files.
- **Metadata maps tensor IDs to file locations**.
- **No built-in compression** (users must generate and store their own compressed representations).

---

## **API Design**

### **1️⃣ Storing and Retrieving Tensors**

#### `put/1`

```elixir
@doc """
Stores a tensor.
- Saves the tensor in the 1D cluster or as a binary file, depending on the shape.
- Updates metadata including shape and, if provided, a raw ND tensor ID.
- Adds tensor (for 1D tensors only) to the IVF cluster and updates HNSW graph.
- Returns tensor_id.
- Note that 1D compressed vector representations of ND tensors must be stored separately.
"""
def put({tensor_id, tensor, metadata}), do: ...
```

#### `put_all/1`

```elixir
@doc """
Stores multiple tensors.
- Saves the tensors in the 1D cluster or as a binary file, depending on the shape.
- Updates metadata including shape and, if provided, a raw ND tensor ID.
- Adds tensor (for 1D tensors only) to the IVF cluster and updates HNSW graph.
- Returns an ordered list of tensor_ids.
- Note that 1D compressed vector representations of ND tensors must be stored separately.
"""
def put_all([{tensor_id, tensor, metadata}, ...]), do: ...
```

#### `get/1`

```elixir
@doc """
Loads a tensor from storage.
- Uses metadata to retrieve the raw tensor ID, if provided.
"""
def get(tensor_id), do: ...
```

#### `get_all/1`

```elixir
@doc """
Loads tensors from storage.
- Uses metadata to retrieve the raw tensor ID, if provided.
"""
def get_all([{tensor_id}, ...]), do: ...
```

#### `query/1`

```elixir
@doc """
Retrieves 1D and ND tensors based on metadata filters (e.g., shape, type).
- Note that ND tensors linked in metadata must be loaded in a separate request by ID.
"""
def query(filters), do: ...
```

#### `query/3`

```elixir
@doc """
Finds similar 1D tensors using ANN + metadata search.
- Searches IVF + HNSW index.
- Returns **only 1D tensors**. Linked ND tensors must be retrieved separately.
- Similarity and distance for ND tensors are roadmap items.
"""
def query(query_vector, k, metadata), do: ...
```

---

### **2️⃣ Memory Management & Caching**

#### `sync/1`

```elixir
@doc """
Ensures a cluster and its graph are in memory and in sync with disk.
- If evicted, the cluster is loaded.
- If dirty, the cluster is written back to disk.
- Otherwise, no-op.
"""
def sync(cluster_id), do: ...
```

#### `sync_lock/1`

```elixir
@doc """
Ensures clusters and graphs are in memory and protected from eviction.
- Calls `sync/1` to ensure data is in memory.
- Marks the cluster as locked, preventing eviction.
"""
def sync_lock([cluster_id, ...]), do: ...
```

#### `sync_unlock/1`

```elixir
@doc """
Writes clusters and graphs back to disk (if dirty) and removes protection from eviction.
"""
def sync_unlock([cluster_id, ...]), do: ...
```

#### `load_cluster/1`

```elixir
@doc """
Loads a cluster into memory, using LRU caching if available.
"""
def load_cluster(cluster_id), do: ...
```

#### `update_cluster/3`

```elixir
@doc """
Updates the HNSW skip-graph when new vectors are inserted.
"""
def update_cluster(graph, cluster, [tensor, ...]), do: ...
```

#### `unload_cluster/1`

```elixir
@doc """
Saves an in-memory cluster to disk.
"""
def unload_cluster(graph, cluster), do: ...
```

#### `rebalance_clusters/1`

```elixir
@doc """
Rebalance (split/merge) one or more clusters if needed.
"""
def rebalance_clusters([cluster_id, ...]), do: ...
```

#### `update_graph/1`

```elixir
@doc """
Add or remove clusters from the global HNSW graph.
"""
def update_graph(add: [cluster, ...], remove: [cluster, ...]) do: ...
```

---

## **Data Storage Format**

### **ND Tensor Storage (Binary Files)**

- **Each ND tensor is stored as a binary file (`Nx.to_binary/1`).**
- **Metadata links tensor IDs to file offsets.**

### **IVF Cluster Storage**

- **1D vectors stored within IVF partitions.**
- **HNSW graphs stored per-cluster (`graph.edges`).**
- **Global HNSW graph links clusters.**

### **Metadata Indexing (ETS or File-Based)**

- **Tracks tensor shape, type, and references.**
- **Maps 1D vectors to ND tensors (if applicable).**

---

## **Query Execution Flow**

1️⃣ **ANN Search:**  

- Query enters **global HNSW graph** → **Finds nearest clusters**.  
- Searches **local HNSW graph within the cluster** → Returns top `k` vectors.  
- **Returns only 1D vectors unless ND tensors are explicitly requested.**

2️⃣ **ND Tensor Retrieval (If Requested):**  

- Uses metadata to check if **ND tensor exists** for a result.  
- **Loads full ND tensor from file storage.**

3️⃣ **Cluster Caching & Rebalancing:**  

- Frequently accessed clusters are **kept in LRU cache**.  
- If a cluster **grows too large**, it **splits**; if too small, it **merges**.
