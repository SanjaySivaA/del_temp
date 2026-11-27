# GPU-Accelerated 2D Delaunay Triangulation in CUDA

A technical summary of the parallel Delaunay Triangulation implementation using explicit CUDA runtime memory management, custom kernels, shared memory optimizations, and stream-event concurrency.

---

## 1. Problem & Core Intuition

### What Problem Does Delaunay Triangulation Solve?
Given a scattered cloud of points on a 2D flat plane, connect every point to its neighbors with straight lines to form a grid of triangles covering the entire region, without any edges crossing.

Many ways exist to connect points into triangles. However, naive connections often create extremely thin, sliver-like "needle" triangles. Sliver triangles cause numerical errors in simulations and graphical rendering glitches.

Delaunay Triangulation guarantees the **"fattest" (most equiangular) triangles possible**.

```
    Bad Triangulation (Slivers)           Delaunay Triangulation (Fatter)
         A --------- B                          A --------- B
          \         /                            \       / /
           \  (x)  /                              \     / /
            \     /                                \   / /
             \   /                                  \ / /
               C                                      C
```

#### The Visual Rule: The "Empty Circle" Condition
Imagine picking any 3 points that form a triangle candidate. Draw the unique circle that passes through all three corner points (its *circumcircle*). 
- If **no other point** in the entire dataset lies strictly inside that circle, the triangle is legally **Delaunay**.
- If any fourth point falls inside that circle, the triangle is "illegal" and must be flipped or rebuilt.

```
       Legal Delaunay Triangle                    Illegal Triangle (Violated)
       
              .---.                                        .---.
            /       \                                    /   P4  \   <-- Point inside!
           A ------- B                                  A ------- B
          / \       / \                                / \       / \
         |   \     /   |                              |   \     /   |
          \   \   /   /                                \   \   /   /
           \    C    /                                  \    C    /
             ' --- '                                      ' --- '
      (Interior is EMPTY)                         (Point P4 breaks the rule)
```

### Real-World Applications
1. **Terrain & Mesh Generation (GIS & Video Games)**: Converting unorganized LiDAR elevation measurements or 3D point scans into continuous, smooth surface meshes for terrain rendering.
2. **Finite Element Analysis (FEA & CFD Physics Simulation)**: Decomposing mechanical parts or fluid domains into triangular elements to solve heat transfer, stress, or fluid equations without numerical singularities caused by skinny triangles.

### Data Formats
The internal engine represents geometry with clean, format-agnostic C/CUDA structures:

```cpp
// 2D Point: Represents a raw coordinate on the flat plane
struct Point2D {
    double x; // X coordinate (double precision for geometric stability)
    double y; // Y coordinate
};

// Triangle: Represents a single triangular surface face by the indices
// of its three corner vertices in counter-clockwise order
struct Triangle {
    int a; // Index of 1st corner vertex (0 to N-1)
    int b; // Index of 2nd corner vertex
    int c; // Index of 3rd corner vertex
};
```

---

## 2. How the Algorithm Works (Step-by-Step)

### High-Level Strategy: Parallel Spatial Grouping
If threads insert points scattered all over the map at random, they access distant memory and fight for the same triangles.

To parallelize efficiently across GPU warps:
1. **Spatial Grouping (Concept)**: We trace a continuous, space-filling snake-like curve across the 2D plane (Z-order / Morton curve). Points that sit next to each other in 2D space are assigned consecutive numbers along this 1D curve.
2. **Parallel Locality**: Adjacent GPU threads process points that are geometric neighbors. When threads walk the mesh to locate their triangles, they walk through the exact same local memory addresses, maximizing L1/L2 cache hits.

```
   2D Plane with Morton Curve Mapping            Points Assigned to Threads
   +-------+-------+                              +-----------------------+
   | (0,1) | (1,1) |   Thread 0 -> Point in cell 0 | Thread 0: (0.1, 0.1)  |
   |   2   |   3   |   Thread 1 -> Point in cell 1 | Thread 1: (0.2, 0.15) |
   +-------+-------+   Thread 2 -> Point in cell 2 | Thread 2: (0.1, 0.8)  |
   |   0   |   1   |   Thread 3 -> Point in cell 3 | Thread 3: (0.8, 0.9)  |
   | (0,0) | (1,0) |                               +-----------------------+
   +-------+-------+                              Coalesced Memory Access!
```

### End-to-End Execution Pipeline
```
[Host RAM: Points] 
       │  (cudaMemcpyAsync)
       ▼
1. GPU Bounding Box Reduction (Shared memory min/max coordinate reduction)
       │
       ▼
2. Morton Spatial Key Generation & Sorting (thrust::sort_by_key on compute_stream)
       │
       ▼
3. Enclosing Super-Triangle Initialization (Single giant triangle covering all points)
       │
       ▼
4. Iterative Point Insertion Waves (Repeats until all points are placed):
   ├── A. Jump-and-Walk Location (Threads step across neighbors to find container)
   ├── B. Conflict-Free Vote (Atomic vote elects one point per triangle)
   ├── C. 1-to-3 Triangle Splitting (Adds new vertices; updates triangle array)
   └── D. Neighbor Pointers Repair (Rewires adjacent triangle edge links)
       │
       ▼
5. 2-Phase Parallel Lawson Edge Flipping (Repeats until empty-circle rule passes):
   ├── A. Propose & Lock Flips (Identifies illegal diagonals; locks pairs with atomicCAS)
   ├── B. Execute Flips (Swaps diagonal edges for locked pairs)
   └── C. Repair Flipped Neighbors (Updates adjoining neighbor references)
       │
       ▼
6. Stream Compaction Filter (Warp-aggregated filter removes super-triangle artifacts)
       │  (cudaMemcpyAsync)
       ▼
[Host RAM: Valid Triangles Output]
```

---

## 3. Kernel Breakdown & Pseudocode

### Host Setup & Stream Management
The host coordinates execution using explicit CUDA runtime calls, pinned host buffers, and asynchronous stream management:

```cpp
// Allocate page-locked (pinned) host memory for high-speed PCIe DMA transfers
Point2D* h_pinned_points;
cudaMallocHost(&h_pinned_points, N * sizeof(Point2D));

// Allocate device memory buffers explicitly
Point2D* d_points;
int3* d_triangles;     // Triangle vertex indices (max 2N + 1)
int3* d_neighbors;     // Adjacency pointers for each of the 3 edges
cudaMalloc(&d_points, (N + 3) * sizeof(Point2D));
cudaMalloc(&d_triangles, max_triangles * sizeof(int3));
cudaMalloc(&d_neighbors, max_triangles * sizeof(int3));

// Create two concurrent streams and synchronization events
cudaStream_t compute_stream, reset_stream;
cudaStreamCreate(&compute_stream);
cudaStreamCreate(&reset_stream);

cudaEvent_t reset_event;
cudaEventCreate(&reset_event);

// Transfer inputs asynchronously to GPU
cudaMemcpyAsync(d_points, h_pinned_points, N * sizeof(Point2D), 
                cudaMemcpyHostToDevice, compute_stream);
```

### Main Device Kernels

#### Kernel 1: Point Location & Voting (`locate_and_vote`)
*Problem solved:* Every uninserted point needs to find which existing triangle currently encloses it, then vote so only one point splits that triangle at a time.

```cpp
__global__ void locate_and_vote(Point2D* points, int3* triangles, int3* neighbors,
                                int* point_tri, int* inserted, int* tri_winner, int N) {
    // Each thread tracks one specific input point
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N || inserted[pt_idx]) return;

    Point2D p = points[pt_idx];
    int t = point_tri[pt_idx]; // Resume walking from last known triangle

    // "Jump-and-Walk": Test which side of each edge the point lies on
    for (int step = 0; step < 10000; ++step) {
        int3 tri = triangles[t];
        int3 nbr = neighbors[t];

        // Geometric orientation tests: Is point 'p' to the left or right of each edge?
        double o0 = orient2d(points[tri.y], points[tri.z], p);
        double o1 = orient2d(points[tri.z], points[tri.x], p);
        double o2 = orient2d(points[tri.x], points[tri.y], p);

        // Step across the edge that separates triangle 't' from point 'p'
        int next_t = -1;
        if (o0 < 0 && nbr.x != -1) next_t = nbr.x;
        else if (o1 < 0 && nbr.y != -1) next_t = nbr.y;
        else if (o2 < 0 && nbr.z != -1) next_t = nbr.z;

        if (next_t != -1) t = next_t; // Step into neighbor triangle
        else break;                   // Point is inside triangle 't'!
    }

    point_tri[pt_idx] = t;

    // Elect the point with the lowest ID to split this triangle this round
    atomicMin(&tri_winner[t], pt_idx);
}
```

#### Kernel 2: 1-to-3 Triangle Splitting (`split_triangles`)
*Problem solved:* Connects the winning interior point to the 3 corners of its enclosing triangle, carving it into 3 smaller triangles.

```cpp
__global__ void split_triangles(int3* triangles, int3* neighbors, int* tri_winner,
                                int* inserted, int* split_base, int* num_triangles, 
                                int prev_count, int N) {
    // Each thread inspects one existing triangle slot
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= prev_count) return;

    int p = tri_winner[t];
    if (p >= N) return; // No point won this triangle; skip

    inserted[p] = 1;

    // Allocate 2 new triangle slots atomically; slot 't' is reused as the 1st child
    int base = atomicAdd(num_triangles, 2);
    split_base[t] = base;

    int3 old_v = triangles[t];
    int3 old_n = neighbors[t];

    int c0 = t, c1 = base, c2 = base + 1;

    // Form 3 counter-clockwise child triangles connecting 'p' to corners
    triangles[c0] = make_int3(p, old_v.y, old_v.z);
    triangles[c1] = make_int3(old_v.z, old_v.x, p);
    triangles[c2] = make_int3(old_v.x, old_v.y, p);

    // Link internal child neighbors and external boundary neighbors
    neighbors[c0] = make_int3(old_n.x, c1, c2);
    neighbors[c1] = make_int3(c2, c0, old_n.y);
    neighbors[c2] = make_int3(c0, c1, old_n.z);
}
```

#### Kernel 3: Propose & Lock Edge Flips (`propose_flips`)
*Problem solved:* Inspects pairs of neighboring triangles sharing an edge. If a corner point falls inside the opposite triangle's circumcircle, threads lock both triangles with atomic compare-and-swap (`atomicCAS`) to safely swap the shared diagonal.

```cpp
__global__ void propose_flips(Point2D* points, int3* triangles, int3* neighbors,
                              int* tri_lock, int* flip_edge, int* flip_partner,
                              int* flip_count, int num_triangles) {
    // Each thread checks one triangle 't'
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int3 nbr = neighbors[t];
    int neighbors_arr[3] = {nbr.x, nbr.y, nbr.z};

    for (int i = 0; i < 3; ++i) {
        int n = neighbors_arr[i];
        if (n <= t || n >= num_triangles) continue; // Avoid deadlock by ordering t < n

        int A = get_opposite_vertex(triangles[t], ...);
        int B = ..., C = ...; // Shared edge vertices
        int D = get_opposite_vertex(triangles[n], B, C);

        // Does point D violate the Empty Circumcircle rule of triangle (A, B, C)?
        if (is_in_circle(points[A], points[B], points[C], points[D]) &&
            is_strictly_convex(points[A], points[B], points[C], points[D])) {
            
            // Acquire mutually exclusive lock on both triangles
            if (atomicCAS(&tri_lock[t], -1, t) == -1) {
                if (atomicCAS(&tri_lock[n], -1, t) == -1) {
                    flip_edge[t] = i;
                    flip_partner[t] = n;
                    flip_partner[n] = t;
                    atomicAdd(flip_count, 1);
                    return; // Locked successfully; ready to flip
                }
                atomicExch(&tri_lock[t], -1); // Rollback lock on conflict
            }
        }
    }
}
```

---

## 4. Shared Memory Usage

### Purpose & Fast On-Chip Storage
Shared memory is high-bandwidth, on-chip SRAM located directly inside each Streaming Multiprocessor (SM). This repository utilizes shared memory for two performance-critical operations:

1. **Bounding Box Parallel Reduction (`compute_bounds_kernel`)**:
   - *Data stored*: Block-local minimum and maximum point coordinates (`s_min_x[256]`, `s_max_x[256]`, etc.).
   - *Why*: Finding coordinate ranges across 1,000,000 points using global memory atomic writes would cause massive memory bus serialization. Keeping intermediate bounds in shared memory allows threads in each block to rapidly reduce 256 coordinates into 1 value on-chip.
2. **Stream Compaction Filter (`filter_triangles_kernel`)**:
   - *Data stored*: Block-level counter `s_count` and global offset `s_offset`.
   - *Why*: Instead of having hundreds of threads execute global memory atomic increments, threads aggregate valid triangle counts in shared memory first.

### Access Pattern & Conflict Minimization
On `sm_80` GPUs, shared memory is divided into 32 independent banks (each 4 bytes wide). If two or more threads in a warp access different addresses within the same bank simultaneously, a *bank conflict* occurs and requests are serialized.

1. **Tree Reduction with Power-of-2 Strides**:
   In `compute_bounds_kernel`, threads combine elements using halving strides:
   ```cpp
   for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
       if (tid < s) {
           s_min_x[tid] = fmin(s_min_x[tid], s_min_x[tid + s]);
       }
       __syncthreads();
   }
   ```
   Sequential thread indices `tid` access contiguous shared memory addresses without stride overlap, guaranteeing conflict-free access.
2. **Warp-Level Hardware Shuffles (`__shfl_down_sync`)**:
   For the final 32 threads within a warp, the kernel switches from shared memory to register-level warp shuffles:
   ```cpp
   for (int offset = 16; offset > 0; offset /= 2) {
       my_min_x = fmin(my_min_x, __shfl_down_sync(0xFFFFFFFF, my_min_x, offset));
   }
   ```
   This eliminates shared memory bank conflicts entirely for the tail reduction, operating at single-cycle register speed.
3. **Warp Aggregation via Ballot Masks**:
   In `filter_triangles_kernel`, threads use `__ballot_sync` and `__popc` to count valid triangles inside registers. Only lane 0 of each warp performs an atomic increment on shared memory `s_count`, cutting shared memory atomic transactions by up to **32x**.

---

## 5. Streams & Concurrency

### Dual-Stream Strategy
The implementation splits workloads across two concurrent streams:
- `compute_stream`: Executes high-throughput geometric kernels (point location, splitting, edge flipping, filtering).
- `reset_stream`: Pre-clears vote buffers, edge locks, and partner indices in the background using fast `cudaMemsetAsync`.

### Overlap & Event Synchronization
Instead of blocking the CPU with `cudaDeviceSynchronize()`, the streams coordinate purely on GPU hardware via `cudaStreamWaitEvent`.

```
Timeline: Overlapped Compute and Memory Buffer Reset
--------------------------------------------------------------------------------------
compute_stream: [ Split Triangles ] ─────────► [ Repair Neighbors ] ──► [ Wait(E1) ] ──► [ Locate & Vote ]
                                                                             ▲
reset_stream:   [ Memset(tri_winner) ] ──► [ Memset(split_base) ] ──► Record(E1)
--------------------------------------------------------------------------------------
                                                                 (Time --->)
```

1. While `compute_stream` finishes neighbor pointer repairs on newly split triangles, `reset_stream` clears `d_tri_winner` and `d_split_base` in parallel.
2. `reset_stream` records event `reset_event`.
3. `compute_stream` executes `cudaStreamWaitEvent(compute_stream, reset_event, 0)`, pausing only at the hardware level until the reset is ready, then immediately launches the next insertion wave.

---

## 6. Time Complexity & Performance

### Algorithmic Complexity

| Phase | Best Case | Average Case | Worst Case (Degenerate) | Space Complexity |
|---|---|---|---|---|
| **Morton Spatial Sort** | $O(N)$ | $O(N \log N)$ | $O(N \log N)$ | $O(N)$ |
| **Point Insertion (Walk)** | $O(N)$ | $O(N \log N)$ | $O(N^2)$ (extreme collinear) | $O(N)$ |
| **Lawson Edge Flipping** | $O(1)$ | $O(N)$ flips | $O(N^2)$ flips (theoretical) | $O(N)$ |
| **Total Memory Footprint** | $O(N)$ | $O(N)$ | $O(N)$ (max $2N + 1$ triangles) | $O(N)$ |

*Note on Euler's Planar Formula*: For any planar triangulation of $N$ points, the maximum possible number of triangles is strictly bounded by $2N + 1$. Thus, memory allocation is statically bounded to $O(N)$ without dynamic reallocations.

### Primary Hardware Bottlenecks
1. **Global Memory Traversal during Mesh Walk**:
   During the jump-and-walk point location, threads chase pointer indices (`neighbors[t]`). While spatial sorting drastically minimizes divergent walks by clustering points, memory latency during triangle hops remains the primary execution cost.
2. **Kernel Launch Round Latency**:
   Edge flipping operates in iterative rounds until no invalid edges remain. As fewer edges require flips near convergence, GPU occupancy decreases, and host-device kernel launch latency becomes noticeable.

---

## 7. References & Lineage

1. **Foundational Edge Flipping**:
   Lawson, C. L. (1977). *Software for $C^1$ surface interpolation*. Mathematical Software III, Academic Press, pp. 161–194. (Introduced the empty-circumcircle criterion and local edge-flipping theorem).
2. **Jump-and-Walk Point Location**:
   Green, P. J., & Sibson, R. (1977). *Computing Dirichlet tessellations in the plane*. The Computer Journal, 21(2), pp. 168–173. (The stochastic/steepest directed walk through triangular adjacencies).
3. **Problem Based Benchmark Suite (PBBS)**:
   Shun, J., Blelloch, G. E., Fineman, J. T., Gibbons, P. B., Kyrola, A., Harsha, P. B. B., & Simhadri, H. V. (2012). *Brief announcement: the Problem Based Benchmark Suite*. SPAA '12. (Defines the standard `pbbs_sequencePoint2d` and `pbbs_triangles` file formats used in this repository).
4. **GPU Delaunay Triangulation Research**:
   Meng, J., & Choi, B. (2014). *gDel2D: High-Performance GPU Delaunay Triangulation for 2D Points*. IEEE Transactions on Visualization and Computer Graphics. (Pioneered spatial sorting and parallel multi-phase edge flips on NVIDIA architectures).
