# NxDB.TensorStore.FileManager

## Overview
`NxDB.TensorStore.FileManager` is responsible for **managing persistent tensor storage** in NxDB. It enables **efficient file-based storage**, supporting **memory-mapped access, cluster-based organization, and GPU-optimized batch processing**.

## Key Responsibilities
- **Efficient File I/O for ANN Storage**
  - Supports **memory-mapped access (`mmap`)** for fast tensor retrieval.
  - Enables **sequential disk access for clustered tensors**.
- **Cluster-Based Storage Organization**
  - Uses **skip-graph hierarchical IVF** for fast lookups.
  - Maintains **per-cluster tensor storage**.
- **GPU-Optimized Batch Processing**
  - Supports **batched reads/writes aligned with GPU-friendly processing**.
  - Uses `Nx.Defn` to optimize distance calculations and retrievals.
- **Automatic Rebalancing**
  - **Monitors cluster sizes** and **splits/merges storage** as needed.
- **LRU-Based Cluster Caching**
  - Keeps **frequently accessed clusters in memory**, reducing disk I/O.

## API Design

### **1. Tensor Storage Management**
#### `store_tensor/2`
```elixir
@doc """
Stores a tensor into the appropriate cluster and writes it to disk.
- Automatically assigns it to the correct **IVF cluster**.
- If the cluster exceeds size limits, triggers rebalancing.
"""
def store_tensor(tensor_id, tensor), do: ...
```

#### `load_tensor/1`
```elixir
@doc """
Loads a tensor from storage using memory-mapped access if available.
- Uses **LRU caching** to keep hot clusters in memory.
"""
def load_tensor(tensor_id), do: ...
```

### **2. Cluster-Based Storage**
#### `store_cluster/2`
```elixir
@doc """
Stores a full IVF cluster to disk.
- Organizes clusters to align with **disk paging & GPU batch sizes**.
"""
def store_cluster(cluster_id, tensors), do: ...
```

#### `load_cluster/1`
```elixir
@doc """
Loads a full cluster into memory for fast ANN search.
- Uses **mmap for large clusters**.
- Fetches from **LRU cache** if recently accessed.
"""
def load_cluster(cluster_id), do: ...
```

### **3. GPU Batch Processing**
#### `batch_store/2`
```elixir
@doc """
Stores multiple tensors in a batch operation.
- Optimized for GPU processing using `Nx.Defn`.
- Ensures writes align with disk page size for efficiency.
"""
def batch_store(tensor_list, cluster_id), do: ...
```

#### `batch_load/1`
```elixir
@doc """
Loads multiple tensors in a batch operation.
- Uses GPU for fast distance computations (`Nx.dot`, `Nx.norm`).
"""
def batch_load(tensor_ids), do: ...
```

### **4. Cluster Rebalancing & LRU Caching**
#### `check_rebalance/1`
```elixir
@doc """
Checks if a cluster needs to be split or merged based on size constraints.
- Uses **dynamic size calculations** based on GPU memory and disk paging.
"""
def check_rebalance(cluster_id), do: ...
```

#### `fetch_from_cache/1`
```elixir
@doc """
Retrieves a cluster from the LRU cache if available.
- Avoids unnecessary disk reads for frequently accessed clusters.
"""
def fetch_from_cache(cluster_id), do: ...
```

## Storage Format
- **Tensors are stored in binary format** (`Nx.to_binary/1`).
- **Clusters are organized as sequential blocks**.
- **Metadata index maps tensor IDs to file offsets**.

## Common Read-Write Use Cases & How This Architecture Addresses Them
### **1. High-Speed kNN Queries (Nearest Neighbor Search)**
**Read Pattern:** Localized sequential reads within an IVF cluster.
**Solution:**
- Uses **IVF for fast cluster selection**.
- Uses **HNSW within clusters** to enable fast neighbor search.
- **Memory-mapped files (`mmap`)** ensure efficient batch retrieval.

### **2. Real-Time Vector Inserts (Feature Store, Streaming Data)**
**Write Pattern:** Frequent small inserts, occasional batch writes.
**Solution:**
- **Append-only writes** keep insert speed high.
- **Automatic rebalancing** prevents cluster overload.
- **Batch updates** are periodically processed via GPU acceleration.

### **3. Large-Scale ANN Query Processing (Batch Search for AI/ML Models)**
**Read Pattern:** Bulk retrievals across multiple clusters.
**Solution:**
- **Parallel GPU-based distance computations (`Nx.Defn`)** accelerate query processing.
- **LRU caching keeps recently queried clusters in memory**, reducing disk I/O.
- **Multi-level IVF indexing (skip-graph structure)** speeds up search over large datasets.

### **4. Anomaly Detection & Outlier Searches**
**Read Pattern:** Sparse, full-dataset scans.
**Solution:**
- **HNSW efficiently narrows the search space**.
- **Batch processing reduces redundant disk reads**.
- **Hybrid CPU-GPU execution ensures scalability.**

### **5. Time-Series Similarity Matching**
**Read Pattern:** Sequential reads across timestamped tensors.
**Solution:**
- **Clusters store temporally adjacent tensors together** for efficient scans.
- **GPU parallelized similarity calculations** improve large-scale temporal searches.
- **LRU cache prefetches likely-to-be-accessed sequences.**

## Next Steps
✅ Implement **memory-mapped tensor retrieval**.  
✅ Integrate **GPU batch processing** via `Nx.Defn`.  
✅ Optimize **LRU cluster caching** for ANN acceleration.  

---
This module ensures that **NxDB.TensorStore scales efficiently** while maintaining **fast ANN search performance**.