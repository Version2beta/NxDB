Here is the **NxDB Key Decisions** document for you to copy into your repository:

---

# **NxDB: Key Decisions (Updated)**

## **1️⃣ Repository & Project Structure**
✅ **Decision:** Keep the project in a personal GitHub repo (not an org) for now.  
✅ **Reasoning:** Faster iteration, easy to transfer later if needed.

---

## **2️⃣ Core Features & Roadmap**
✅ **Decision:** NxDB supports **storing ND tensors** but **only indexes 1D tensors (vectors) for ANN search** initially.  
✅ **Reasoning:** ANN methods (HNSW, IVFFlat) work best on 1D vectors. Future roadmap includes **KD-Trees, R-Trees for ND indexing**.

✅ **Decision:** **Memory-mapped storage for tensors** (fast reads, no serialization overhead).  
✅ **Reasoning:** Enables **fast persistence without needing to load everything into RAM**.

✅ **Decision:** **HNSW for ANN search**, initially **RAM-only, later hybrid disk-based**.  
✅ **Reasoning:** Balances **speed, memory efficiency, and scalability**.

✅ **Decision:** **Metadata stored in ETS** (fast ID lookups).  
✅ **Reasoning:** Elixir’s ETS is **O(1) for lookups**, making it ideal for **ID → storage mappings**.

✅ **Decision:** Use a **Skip-Graph-Like Hierarchical IVF** for cluster organization.  
✅ **Reasoning:** Multi-level IVF improves **scalability and query efficiency** by enabling hierarchical refinement.

✅ **Decision:** Use **HNSW within each IVF cluster** instead of a single global HNSW graph.  
✅ **Reasoning:** Keeps **local searches efficient** while avoiding **global graph bottlenecks**.

✅ **Decision:** Implement **an HNSW graph of clusters** for fast high-level cluster selection.  
✅ **Reasoning:** Enables **fast ANN search across clusters**, reducing exhaustive scans.

✅ **Decision:** Introduce **automatic cluster rebalancing** (splitting/merging based on size).  
✅ **Reasoning:** Prevents clusters from **becoming too large or too small**, keeping query performance optimal.

✅ **Decision:** Dynamically **calculate cluster sizes** based on **GPU memory and disk paging constraints**.  
✅ **Reasoning:** Ensures **batch operations are GPU-friendly** and **disk I/O is optimized**.

✅ **Decision:** Use **Nx + EXLA/Torchx for GPU acceleration** of ANN computations.  
✅ **Reasoning:** GPU-based **dot product, distance calculations, and batch processing** will provide **10-100x faster queries**.

✅ **Decision:** Implement **LRU-based cluster caching**, treating clusters as **memory pages** that swap efficiently.  
✅ **Reasoning:** Reduces **disk I/O overhead** while **ensuring hot clusters remain in memory**.

---

## **3️⃣ Performance Benchmarks & Competitive Positioning**
✅ **Decision:** **Benchmark NxDB vs. FAISS, Milvus, Annoy, Pinecone, PGVector.**  
✅ **Reasoning:** NxDB should be competitive while remaining **Elixir-native, Nx-optimized**.

✅ **Decision:** Benchmark **Hybrid IVF + HNSW + LRU Caching vs. alternative ANN architectures** (IVF-only, HNSW-only, FAISS, Milvus).  
✅ **Reasoning:** Ensure **best scalability** while keeping insert/query balance **competitive with state-of-the-art ANN solutions**.

✅ **Decision:** Include **GPU-accelerated benchmarks** using `Nx + EXLA/Torchx`.  
✅ **Reasoning:** Measure **real-world performance improvements from GPU acceleration** on **query latency, batch processing speed, and memory efficiency**.

✅ **Decision:** Evaluate **dynamic cluster rebalancing impact on ANN query performance**.  
✅ **Reasoning:** Ensure that **automatic splitting/merging does not degrade retrieval speed** under heavy insert loads.

✅ **Decision:** Compare **LRU-based cluster caching vs. on-demand disk retrieval**.  
✅ **Reasoning:** Quantify **performance gains from keeping hot clusters in memory** vs. direct disk access.

✅ **Decision:** Test **multi-level IVF (skip-graph-like structure) vs. flat IVF**.  
✅ **Reasoning:** Validate **scalability improvements** and **query speed optimizations** introduced by hierarchical IVF.

✅ **Predicted Performance:**
| **Test** | **NxDB Expected Performance** | **Comparison (FAISS, Pinecone, etc.)** |
|----------|------------------------------|---------------------------------------|
| **Insert 1M 1D tensors (128D, SSD)** | **2-5s** | **FAISS: ~4.5s, Milvus: ~5.2s** |
| **Retrieve 1 tensor by ID (RAM)** | **<100μs** | **FAISS: ~80μs, Milvus: ~90μs** |
| **Batch retrieve 10K tensors (RAM)** | **<30ms** | **FAISS: ~25ms, Milvus: ~28ms** |
| **Single ANN search (RAM, HNSW)** | **<3ms** | **FAISS: ~2.5ms, Milvus: ~3ms** |
| **Single ANN search (Hybrid SSD)** | **<10ms** | **FAISS: ~8ms, Milvus: ~12ms** |
| **Batch ANN search (10K, RAM)** | **<300ms** | **FAISS: ~280ms, Milvus: ~290ms** |
| **Batch ANN search (10K, GPU)** | **<100ms** | **FAISS GPU: ~90ms, Pinecone GPU: ~110ms** |

---

## **4️⃣ API & Integration Strategy**
✅ **Decision:** NxDB API should **follow Nx’s function chaining style**.  
✅ **Reasoning:** Makes **composability easy** for ML/AI workloads.

✅ **Decision:** Future **REST/gRPC API will be built on `NxDB` core**.  
✅ **Reasoning:** Keeps the system modular.

✅ **Decision:** Use **`Nx.Defn` for ANN computations**, ensuring **automatic hardware acceleration**.  
✅ **Reasoning:** Simplifies **GPU acceleration without requiring manual CUDA/OpenCL programming**.

✅ **Decision:** Optimize Cluster Sizes Dynamically  
✅ **Reasoning:** Balances **VRAM utilization, disk paging efficiency, and GPU batch processing**.

✅ **Decision:** Ensure Memory-Mapped Storage (`mmap`) Supports ANN Workloads  
✅ **Reasoning:** Minimizes overhead for **large-scale vector search and batch retrievals**.

✅ **Decision:** Hybrid Storage Strategy for ANN Queries  
✅ **Reasoning:** **Queries use memory-cached clusters first**, then **disk-based clusters**, preventing slowdowns.

---

## **5️⃣ Next Steps**
✅ **Immediate Priorities:**
1. **Implement `NxDB.TensorStore`** (memory-mapped storage).
2. **Develop `NxDB.Metadata`** (ETS-based indexing).
3. **Implement `NxDB.AnnIndex`** (HNSW, RAM-first, hybrid later).
4. **Run benchmarks & optimize.**
5. **Develop `NxDB.TensorStore.FileManager` for optimized ANN storage, leveraging LRU caching and GPU acceleration.**
6. **Refine hierarchical IVF-HNSW integration with automatic rebalancing.**
7. **Ensure Nx-based GPU optimizations align with real-world hardware constraints.**

---

🚀 **This updated document consolidates all key decisions for NxDB, ensuring it remains scalable, efficient, and GPU-accelerated.** Would you like any refinements or additional sections?