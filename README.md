# NxDB: An Elixir-Native Tensor Database

## Overview

NxDB is a high-performance, Elixir-native tensor database designed to handle massive datasets efficiently. It leverages **ETS for metadata indexing**, **a hybrid disk + in-memory ANN index (HNSW)** for fast approximate nearest neighbor search, and **memory-mapped storage** for scalable tensor persistence. NxDB supports **storing Nx tensors of any dimensionality**, while **ANN indexing is  limited to 1D tensors (vectors)**, with future support planned for **multi-dimensional indexing**.

## Goals

- **Elixir-First:** Fully integrated with Elixir and Nx for efficient tensor operations.
- **Scalable Storage:** Handle datasets too large for RAM via memory-mapped disk storage.
- **Fast Approximate Search:** Implement HNSW for high-speed ANN queries on 1D vectors.
- **Hybrid Architecture:** Efficiently manage large-scale tensor data with an in-memory + disk index.
- **GPU Acceleration:** Support EXLA/Torchx for high-performance tensor computations.

## Relevance & Benefits to the Elixir + Nx Ecosystem

Elixir has seen **growing adoption in AI, ML, and high-performance computing**, particularly with **Nx** (Numerical Elixir). However, **no dedicated Elixir-native tensor database exists today**. NxDB fills this gap by providing:

1. **Bridging AI & Data Storage in Elixir** → Nx enables powerful tensor computations, but Elixir lacks an efficient way to **store, query, and search high-dimensional tensors**. NxDB enables **persistent, scalable tensor storage** optimized for ANN search on 1D vectors.
2. **Scaling Beyond Memory Limits** → Traditional in-memory solutions are limited by RAM. NxDB’s **memory-mapped storage** and future **hybrid disk-based indexing** allow Elixir applications to **scale tensor workloads** efficiently.
3. **Unlocking Fast, GPU-Accelerated Similarity Search** → With built-in support for **HNSW indexing and EXLA/Torchx acceleration**, NxDB makes **vector search performant and parallelizable** on GPUs.
4. **Optimized for Retrieval-Augmented Generation (RAG)** → Supports **tensor search for AI-generated responses**, making it a viable solution for **retrieval-augmented generation workloads** directly within Elixir.
5. **Nx Tensors Stored Natively (Zero Serialization Overhead)** → Unlike other tensor databases that require **manual serialization/deserialization**, NxDB stores **raw Nx tensors**, avoiding unnecessary encoding/decoding steps.
6. **Pipeline-Friendly API** → Inspired by Nx’s **function chaining**, allowing composable query workflows, e.g.:

```elixir
query
|> NxDB.search(k: 10)
|> NxDB.filter_by(:category, "electronics")
|> NxDB.sort_by(:score)
```

🚀 **NxDB wants to be a tensor database that enables scalable, high-performance AI applications within the Elixir ecosystem.**

## Installation

When available in Hex ([TO-DO: Publish in Hex](https://hex.pm/docs/publish)), the package can be installed
by adding `nxdb` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:nxdb, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc) and published on [HexDocs](https://hexdocs.pm). Once published, the docs can be found at [https://hexdocs.pm/nxdb](https://hexdocs.pm/nxdb).

## Architecture

### **Components**

1. **Tensor Store (Memory-Mapped Storage)**

   - Stores Nx tensors of any dimensionality efficiently on disk without loading the entire dataset into RAM.
   - Supports direct memory mapping for fast, on-demand tensor access.

2. **ANN Index (HNSW)**

   - Implements **Hierarchical Navigable Small World (HNSW)** for fast approximate nearest neighbor search.
   - Initially supports **only 1D tensors (vectors)** for ANN indexing.
   - Can run fully **in-memory**, or as a **hybrid RAM + disk-based** model.

3. **Metadata Index (ETS)**

   - Stores vector metadata and tensor file offsets for quick lookups.
   - Keeps a lightweight mapping of **tensor ID → location in memory-mapped storage**.

4. **Elixir API (**`NxDB`)

   - Provides a unified interface to interact with all components.
   - Ensures a simple and efficient API for storing, querying, and managing tensors.

## File Structure

```sh
nxdb/
├── lib/
│   ├── nxdb.ex              # Main API module
│   ├── nxdb/tensor_store.ex # Handles memory-mapped tensor storage
│   ├── nxdb/metadata.ex     # ETS-based metadata store
│   ├── nxdb/ann_index.ex    # HNSW-based ANN index
│   ├── nxdb/config.ex       # Configuration management
│   ├── nxdb/server.ex       # GenServer wrapper (for future clustering)
│
└── test/
    ├── nxdb_test.exs
    ├── nxdb/tensor_store_test.exs
    ├── nxdb/metadata_test.exs
    ├── nxdb/ann_index_test.exs
```

## Roadmap

### **1️⃣ Tensor Store (Memory-Mapped Storage)**

- **Store Tensors** → Insert Nx tensors (any shape) into the database.
- **Batch Processing** → Efficiently handle large-scale inserts.
- **Streaming Support** → Enable efficient retrieval and partial loading of large datasets.

### **2️⃣ ANN Index (HNSW for Approximate Nearest Neighbor Search)**

- **Indexing (HNSW)** → Build and update an HNSW, disk-based+in-memory index for fast queries.
- **Query 1D Tensors (Vectors)** → Retrieve tensors by ANN search (1D only).
- **GPU Acceleration** → Offload search operations to EXLA/Torchx.

### **3️⃣ Metadata Index (ETS for Fast Lookups)**

- **Store Tensors** → Maintain tensor ID → location mappings.
- **Query Tensors** → Retrieve tensors by ID.
- **Delete Tensors** → Remove tensors and update indexes.
- **Config Management** → Enable tunable parameters (memory limits, index strategy, PQ).

### **4️⃣ Elixir API (**`NxDB`)

- **Insert API** → Provide a unified function to insert tensors into the database.
- **Query API** → Unified function for searching tensors by ID or ANN search.
- **Delete API** → Function to remove tensors while keeping indexes consistent.
- **Batch Operations** → Allow bulk inserts and retrievals efficiently.
- **Future REST/gRPC API** → Build on top of `NxDB` when needed.

## Next Steps

1. **Implement `NxDB.TensorStore`** → Memory-map Nx tensors for fast access.
2. **Develop `NxDB.Metadata`** → ETS-based metadata tracking.
3. **Implement `NxDB.AnnIndex`** → Hybrid disk-based+in-memory HNSW nearest neighbor search.
4. **Integrate `NxDB` API** → Unified interface for storing, querying, and managing tensors.

## Future Enhancements

- **Multi-Dimensional Tensor Indexing** → Implement KD-Trees, R-Trees, and/or tensor hashing.
- **Streaming Data Ingestion** → Efficient handling of real-time tensor updates.
- **Quantization Support** → Reduce memory usage for massive datasets.
- **Multi-Node Scaling** → Transition to a clustered NxDB architecture when required.
- **Cloud Deployment Support** → Optimize for distributed and serverless environments.

## **📌 Benchmarking Plan for NxDB Performance**  

To verify the predicted performance, we need a structured **benchmarking plan** covering storage, ANN search, and metadata lookup performance. Below is a step-by-step approach.

### **1️⃣ Benchmarking Environment**

#### **Hardware Setup**  

- **CPU:** Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz  
- **RAM:** 256GB+ (for memory-mapped optimizations)  
- **Disk:** U.2 SSD (for high I/O performance)  
- **GPU (Optional):** NVIDIA A4000+ (for GPU-accelerated ANN search via EXLA/Torchx)  

#### **Software Setup**

- **Elixir Version:** `>= 1.14`  
- **NxDB Version:** Latest build  
- **Benchmarking Tool:** `benchee` for Elixir  
- **Dataset:**  
  - **Randomly generated Nx tensors**  
  - **Pretrained embeddings (e.g., OpenAI, Sentence Transformers)**  
  - **Multiple tensor dimensions: `{128}`, `{512}`, `{768}` (BERT), `{2048}` (ResNet)**  

---

### **2️⃣ Benchmark: Memory-Mapped Storage Performance**  

#### **Objective: Storage Performance**  

- Measure **insertion, retrieval, and batch query speeds** for Nx tensors.  

#### **Storage Scenarios to Test:**  

| Test | Expected Performance |
|------|----------------------|
| **Insert 1M 1D tensors (128D, float32)** | **2s – 5s** |
| **Insert 1M ND tensors (3D, float32, 3x224x224)** | **3s – 10s** |
| **Retrieve 1 tensor by ID (1D, memory-mapped)** | **< 100μs** |
| **Retrieve 1 tensor by ID (ND, memory-mapped)** | **< 3ms** |
| **Batch retrieve 10K tensors (1D, memory-mapped)** | **< 30ms** |

---

### **3️⃣ Benchmark: ANN Search Performance (HNSW)**  

#### **Objective: ANN Search Performance**  

- Measure **ANN search speed** for **in-memory vs hybrid disk-based indexing**.  

#### **ANN Search Scenarios to Test:**  

| Test | Expected Performance |
|------|----------------------|
| **Search 1 vector (128D, HNSW in-memory)** | **< 3ms** |
| **Search 1 vector (128D, HNSW hybrid disk-based)** | **< 10ms** |
| **Batch search 10K vectors (128D, HNSW in-memory)** | **< 300ms** |
| **Batch search 10K vectors (128D, GPU-Accelerated)** | **< 100ms** |

---

### **4️⃣ Benchmark: Metadata Index Performance (ETS)**  

#### **Objective: Metadata Index Performance**

- Measure **metadata insert, lookup, and batch query speeds**.  

#### **Scenarios to Test: Metadata Index Performance**  

| Test | Expected Performance |
|------|----------------------|
| **Insert 1M metadata entries** | **< 3s** |
| **Retrieve 1 metadata entry** | **< 100μs** |
| **Batch retrieve 10K metadata entries** | **< 30ms** |

---

### **5️⃣ Benchmark: Mixed Workload**  

#### **Objective: Mixed Workload**

- Simulate real-world **insertion, ANN search, and metadata retrieval combined**.  
- Test **parallel requests** to evaluate concurrency.  

#### **Scenarios to Test: Mixed Workload**  

| Test | Expected Performance |
|------|----------------------|
| **Insert + Search + Metadata lookup (1M operations, mixed)** | **< 7s** |
| **Concurrent ANN queries (10K queries, multi-threaded)** | **< 300ms** |

---

### **📌 Summary of Benchmarking Plan**  

| Benchmark | Tests | Expected Performance |
|-----------|------|----------------------|
| **Storage Performance** | Insert, retrieve, batch load | **1GB/s – 3.5GB/s, <100μs retrieval** |
| **ANN Indexing Performance** | Search, batch search | **< 3ms (RAM), < 10ms (Hybrid), < 100ms (GPU)** |
| **Metadata Index Performance** | Insert, lookup, batch query | **< 100μs (single), < 30ms (10K batch)** |
| **Mixed Workload** | Insert + ANN Search + Metadata | **< 7s (1M ops), < 300ms (concurrent)** |
