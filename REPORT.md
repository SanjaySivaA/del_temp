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

### The Mental Model: Building a Mesh from Scratch
If you try to connect 10,000 points into triangles all at once, you face a combinatorial nightmare: edges cross each other, holes form, and tracking which points can connect becomes difficult.

Instead of building connections from a blank slate, this algorithm uses an approach called **Incremental Insertion with Edge Flipping**:

1. **Create an Initial Sandbox**: We start with one giant triangle, called the *super-triangle*, that is large enough to contain every single input point inside it.
2. **Drop Points into Triangles**: Whenever you drop a point inside an existing triangle, you can connect that point to the triangle's three corner vertices. One triangle cleanly splits into three smaller triangles.
3. **Fix Bad Connections (Edge Flips)**: Splitting triangles into smaller pieces creates skinny, needle-like triangles. Once all points are inside the mesh, we look at pairs of neighboring triangles. If swapping their shared diagonal edge creates fatter triangles that satisfy the empty circle rule, we flip that edge. We keep flipping until every edge in the mesh is valid.
4. **Remove the Sandbox**: We erase the three artificial corner vertices of the original giant triangle and discard any triangles attached to them. What remains is the exact Delaunay Triangulation of your original points.

Running this process sequentially on a CPU is simple, but running it across thousands of parallel GPU threads requires addressing two challenges:
- **Spatial locality**: How to prevent threads from thrashing memory when searching for triangles.
- **Race conditions**: How to stop multiple threads from modifying the same triangle or edge simultaneously.

Here is how each phase solves these problems on the GPU.

---

### Phase 1: Spatial Ordering for Warp Locality
If thread 0 processes a point at coordinate (0.01, 0.01) while thread 1 processes a point at (0.99, 0.99), the two threads will walk in opposite directions across the 2D plane. They will access entirely different regions of GPU global memory, causing memory divergence and severe cache thrashing.

To make threads cooperate:
1. We compute the bounding box (minimum and maximum X and Y coordinates) of all points using a fast parallel reduction kernel.
2. We map each 2D point coordinate $(x, y)$ to an integer along a 1D space-filling curve called a **Morton curve** (or Z-order curve). This curve weaves through 2D space like a recursive snake.
3. Points that sit close to each other in 2D space receive numbers that are close to each other along the 1D curve.
4. We sort the points along this curve using `thrust::sort_by_key`.

Now, threads in the same 32-thread GPU warp process points located in the exact same physical neighborhood. When they search for triangles in memory, they read from the exact same cache lines.

```
      2D Grid to 1D Morton Curve Mapping              Threads in Warp Access Same Cache Line
      +-------+-------+-------+-------+
      |  10   |  11   |  14   |  15   |               Thread 0: Point at (0.05, 0.05) [Key 0]
      +-------+-------+-------+-------+               Thread 1: Point at (0.08, 0.12) [Key 0]
      |   8   |   9   |  12   |  13   |               Thread 2: Point at (0.15, 0.08) [Key 1]
      +-------+-------+-------+-------+               Thread 3: Point at (0.22, 0.18) [Key 2]
      |   2   |   3   |   6   |   7   |
      +-------+-------+-------+-------+               Nearby points mean neighboring threads
      |   0   |   1   |   4   |   5   |               read the exact same triangle memory!
      +-------+-------+-------+-------+
```

---

### Phase 2: Setting up the Super-Triangle Sandbox
Before inserting points, the GPU initializes the canvas with a single triangle:
- Point $N$ at $(-100.0, -100.0)$
- Point $N+1$ at $(100.0, -100.0)$
- Point $N+2$ at $(0.0, 100.0)$

Because all input points are normalized to $[0.1, 0.9] \times [0.1, 0.9]$, every point is deep inside this initial triangle. 

We maintain two primary arrays in GPU global memory:
- `triangles[t]`: Stores the three vertex indices $(v_0, v_1, v_2)$ for triangle $t$.
- `neighbors[t]`: Stores the indices of the three neighboring triangles that share an edge with triangle $t$. If an edge lies on the outer boundary of the mesh, its neighbor index is set to $-1$.

At initialization, `triangles[0] = {N, N+1, N+2}` and `neighbors[0] = {-1, -1, -1}`.

---

### Phase 3: Inserting Points in Parallel Waves
We cannot insert all points in one parallel step. If five threads try to split the same triangle simultaneously, they will overwrite each other's writes and corrupt the mesh. 

Instead, insertion proceeds in **iterative waves** until every point is placed:

```
                          Point Insertion Wave Loop
                          
  [ Uninserted Points ] ──► Step A: Walk Mesh to Locate Containing Triangle
                                   │
                                   ▼
                            Step B: Atomic Vote
                            (One point wins per triangle)
                                   │
                                   ▼
                            Step C: 1-to-3 Triangle Split
                            (Winning triangles become 3 smaller triangles)
                                   │
                                   ▼
                            Step D: Repair Neighbor Pointers
                                   │
                                   ▼
                           All points placed? ──(No)──► Repeat Wave
                                   │ (Yes)
                                   ▼
                         Proceed to Edge Flipping
```

#### Step A: Mesh Walking (Jump-and-Walk)
Each thread is assigned one point. To find which triangle contains that point:
- The thread starts at a known triangle (initially triangle 0).
- It runs a 2D orientation test on each of the triangle's three edges. The orientation test calculates the signed area of a small triangle formed by the edge and the test point. A positive value means the point is to the left of the directed edge; a negative value means it is to the right.
- If the point is to the right of an edge, it lies outside that edge. The thread follows the `neighbors` pointer across that edge into the adjacent triangle.
- The thread repeats this hop until the point is inside all three edges of the current triangle.
- Because points are sorted spatially and remember their containing triangle from previous waves, walking takes only a few hops per wave.

#### Step B: Voting for Triangle Ownership
Multiple uninserted points often land inside the same triangle. To avoid write collisions:
- Every point inside triangle $T$ attempts an atomic minimum on a shared voting slot: `atomicMin(&tri_winner[T], point_id)`.
- The point with the smallest ID wins the right to split triangle $T$ in this wave.
- The other points do nothing and remain uninserted, waiting for the next wave.

#### Step C: 1-to-3 Triangle Splitting
For every triangle $T$ that had a winner:
- The thread allocates two new triangle slots in global memory using `atomicAdd(num_triangles, 2)`.
- The original slot $T$ is reused as Child 0. The two newly allocated slots become Child 1 and Child 2.
- Point $P$ is placed inside, and edges are drawn from $P$ to the three corners of $T$.
- One triangle is now replaced by three smaller triangles, all correctly wound in counter-clockwise order.

```
             Before Split                               After 1-to-3 Split
                  v2                                            v2
                 /  \                                          / | \
                /    \                                        /  |  \
               /  P   \        ==== 1-to-3 Split ====>       /  (c0) \
              /        \                                    /    |    \
             /          \                                  / c1  P  c2 \
            v0 -------- v1                                v0 ---───--- v1
```

#### Step D: Neighbor Pointer Repair
When triangle $T$ splits into three children, the surrounding triangles that bordered $T$ still hold obsolete pointers pointing to the old single triangle $T$.
- A repair kernel runs across all triangles.
- Each thread checks whether its neighbors were split during this wave. If so, it looks up the split record to find which of the three children now shares that specific edge, and updates its neighbor pointer accordingly.

This wave loop repeats until all points have been inserted.

---

### Phase 4: Fixing Illegal Edges (Parallel Lawson Edge Flipping)
After insertion finishes, all points exist within a continuous triangular mesh. However, many triangles are skinny and irregular. We now convert this mesh into a Delaunay mesh by flipping edges that violate the empty circle rule.

#### What is an Edge Flip?
Look at two neighboring triangles, $ABC$ and $BDC$, that share edge $BC$. Together they form a four-sided quadrilateral $ABDC$:

```
        Before Flip (Illegal Edge BC)           After Flip (Legal Edge AD)
                     A                                       A
                   / | \                                   /   \
                  /  |  \                                 /     \
                 B   |   C        === Edge Flip ===>     B ────── C
                  \  |  /                                 \     /
                   \ | /                                   \   /
                     D                                       D
            (Skinny triangles)                      (Fatter triangles)
```

- We test whether vertex $D$ lies inside the circumcircle of triangle $ABC$.
- If $D$ is inside the circle, edge $BC$ violates the Delaunay rule.
- We erase edge $BC$ and connect vertex $A$ to vertex $D$ instead.
- This single swap replaces two skinny triangles with two wider, more equiangular triangles.

#### How We Flip in Parallel on the GPU
If two threads try to flip adjacent edges at the same time, they will access the same vertices and scramble the mesh topology. We avoid race conditions using a **2-Phase Lock and Execute mechanism**:

```
                  Parallel Edge Flipping Round Loop
                  
  [ All Triangles ] ──► Phase 1: Propose Flips & Acquire Locks
                                 - Thread t checks neighbor n (where t < n)
                                 - Checks empty-circle rule and convexity
                                 - Locks both t and n using atomicCAS
                                 │
                                 ▼
                        Phase 2: Execute Locked Flips
                                 - Winning thread swaps diagonal edge
                                 - Updates 2 triangle slots and neighbor links
                                 │
                                 ▼
                        Phase 3: Repair Surrounding Neighbors
                                 - Adjacent triangles update neighbor pointers
                                 │
                                 ▼
                     Any flips performed? ──(Yes)──► Next Round
                                 │ (No)
                                 ▼
                   Mesh is Guaranteed Delaunay!
```

1. **Phase 1: Propose and Lock**:
   - Each thread inspects triangle $t$ and its neighbor $n$.
   - To prevent deadlocks, thread $t$ only considers neighbors with higher indices ($t < n$).
   - If edge $(t, n)$ violates the empty circle rule and the four points form a strictly convex quadrilateral, thread $t$ attempts to lock both triangles:
     ```cpp
     if (atomicCAS(&tri_lock[t], -1, t) == -1) {
         if (atomicCAS(&tri_lock[n], -1, t) == -1) {
             // Successfully locked both triangles: mark edge for flipping
         } else {
             atomicExch(&tri_lock[t], -1); // Roll back if neighbor was busy
         }
     }
     ```
   - Only non-conflicting pairs are locked in any given round.

2. **Phase 2: Execute Flips**:
   - Each thread that successfully locked a pair updates the vertex indices of triangles $t$ and $n$ to draw the new diagonal.
   - It updates the internal neighbor pointers between $t$ and $n$.

3. **Phase 3: Repair Neighbor References**:
   - Surrounding triangles whose shared edges moved during the flip update their neighbor pointers to point to the correct triangle.

We repeat this three-step round until zero flips occur across the entire mesh (typically 10 to 30 rounds). Lawson's theorem guarantees that this process always terminates, and when zero flips remain, every triangle in the mesh satisfies the empty circle rule.

---

### Phase 5: Erasing the Super-Triangle and Compacting Output
The triangulation is now mathematically Delaunay, but it still contains the three artificial super-triangle vertices $(N, N+1, N+2)$ and their attached boundary triangles.

To finalize the output:
1. A stream compaction kernel inspects each triangle.
2. If any of the triangle's three corners has an index $\ge N$, that triangle is connected to the super-triangle and is discarded.
3. If all three corners are $< N$, the triangle is a genuine part of the original point set. The thread remaps the internal sorted point indices back to the original input point IDs.
4. Using warp-level aggregation, valid triangles are written into a contiguous output buffer and transferred back to host RAM.

---

### Complete High-Level Execution Pseudocode

```
Algorithm: GPU-Accelerated 2D Delaunay Triangulation

Input : Array of N 2D points on host
Output: Array of Delaunay triangles (each a triple of point indices)

Host Execution:
1. Allocate pinned host buffers and device memory via cudaMalloc.
2. Copy input points to device asynchronously on compute_stream.
3. Launch compute_bounds_kernel: reduce min/max coordinates using shared memory.
4. Launch compute_morton_keys_kernel: map (x, y) coordinates to 1D Morton curve keys.
5. Sort (keys, permutation_indices) using thrust::sort_by_key on compute_stream.
6. Launch unpack_sorted_points: arrange points in sorted order.
7. Launch init_super_triangle: write vertices (N, N+1, N+2) and initial triangle 0.

Point Insertion Loop:
8. While (uninserted points remain):
     a. On reset_stream: clear tri_winner votes and split_base buffers using cudaMemsetAsync.
     b. Trigger compute_stream to proceed via cudaStreamWaitEvent.
     c. Launch locate_and_vote:
          - Each thread walks across neighbor pointers using 2D orientation tests.
          - Upon reaching containing triangle t, cast atomic vote: atomicMin(&tri_winner[t], point_id).
     d. Launch split_triangles:
          - Winning triangles allocate 2 new slots via atomicAdd(num_triangles, 2).
          - Split container into 3 child triangles and update internal links.
     e. Launch repair_neighbors:
          - Neighbor triangles re-point their adjacency references to the correct child.

Edge Flipping Loop:
9. Loop:
     a. On reset_stream: clear tri_lock, flip_edge, and flip_count buffers.
     b. Trigger compute_stream via cudaStreamWaitEvent.
     c. Launch propose_flips:
          - Inspect shared edges where neighbor index n > t.
          - If opposite vertex fails empty-circle test, lock t and n using atomicCAS.
     d. Read flip_count back to host. If flip_count == 0, break loop.
     e. Launch execute_flips:
          - Locked pairs swap the shared diagonal edge and re-link internal neighbors.
     f. Launch repair_flipped_neighbors:
          - Adjacent triangles update neighbor references to reflect flipped diagonals.

Output Compaction:
10. Launch filter_triangles_kernel:
      - Discard triangles containing super-triangle vertices (index >= N).
      - Remap valid triangle vertices back to original input point IDs.
      - Aggregate counts using warp ballots and write to contiguous output array.
11. Copy final triangles buffer to host RAM asynchronously and synchronize.
12. Free all host and device memory buffers.
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
   Lawson, C. L. (1977). *Software for $C^1$ surface interpolation*. Mathematical Software III, Academic Press, pp. 161-194. (Introduced the empty-circumcircle criterion and local edge-flipping theorem).
2. **Jump-and-Walk Point Location**:
   Green, P. J., & Sibson, R. (1977). *Computing Dirichlet tessellations in the plane*. The Computer Journal, 21(2), pp. 168-173. (The stochastic/steepest directed walk through triangular adjacencies).
3. **Problem Based Benchmark Suite (PBBS)**:
   Shun, J., Blelloch, G. E., Fineman, J. T., Gibbons, P. B., Kyrola, A., Harsha, P. B. B., & Simhadri, H. V. (2012). *Brief announcement: the Problem Based Benchmark Suite*. SPAA '12. (Defines the standard `pbbs_sequencePoint2d` and `pbbs_triangles` file formats used in this repository).
4. **GPU Delaunay Triangulation Research**:
   Meng, J., & Choi, B. (2014). *gDel2D: High-Performance GPU Delaunay Triangulation for 2D Points*. IEEE Transactions on Visualization and Computer Graphics. (Pioneered spatial sorting and parallel multi-phase edge flips on NVIDIA architectures).
