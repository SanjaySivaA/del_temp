#include "delaunay.cuh"
#include "predicates.cuh"
#include <iostream>
#include <climits>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while (0)

// -----------------------------------------------------------------------------
// CUDA Helper & Utility Kernels
// -----------------------------------------------------------------------------

template <typename T>
__global__ void fill_kernel(T* data, T val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

// Atomic min and max for double precision using atomicCAS
__device__ inline void atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__device__ inline void atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// Bounding box reduction kernel with shared memory and bank-conflict avoidance
__global__ void compute_bounds_kernel(
    const Point2D* points,
    double* d_min_x, double* d_max_x,
    double* d_min_y, double* d_max_y,
    int n
) {
    __shared__ double s_min_x[256];
    __shared__ double s_max_x[256];
    __shared__ double s_min_y[256];
    __shared__ double s_max_y[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double my_min_x = 1e30, my_max_x = -1e30;
    double my_min_y = 1e30, my_max_y = -1e30;

    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        Point2D p = points[i];
        my_min_x = fmin(my_min_x, p.x);
        my_max_x = fmax(my_max_x, p.x);
        my_min_y = fmin(my_min_y, p.y);
        my_max_y = fmax(my_max_y, p.y);
    }

    s_min_x[tid] = my_min_x;
    s_max_x[tid] = my_max_x;
    s_min_y[tid] = my_min_y;
    s_max_y[tid] = my_max_y;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = fmin(s_min_x[tid], s_min_x[tid + s]);
            s_max_x[tid] = fmax(s_max_x[tid], s_max_x[tid + s]);
            s_min_y[tid] = fmin(s_min_y[tid], s_min_y[tid + s]);
            s_max_y[tid] = fmax(s_max_y[tid], s_max_y[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMinDouble(d_min_x, s_min_x[0]);
        atomicMaxDouble(d_max_x, s_max_x[0]);
        atomicMinDouble(d_min_y, s_min_y[0]);
        atomicMaxDouble(d_max_y, s_max_y[0]);
    }
}

// Normalizes points to [0.1, 0.9] and computes 32-bit Morton code keys
__global__ void normalize_and_morton_keys(
    const Point2D* in_points,
    PointWithIdx* out_points,
    uint32_t* cell_keys,
    double min_x, double max_x,
    double min_y, double max_y,
    int n, int padded_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_n) return;

    if (idx < n) {
        Point2D p = in_points[idx];
        double span_x = max_x - min_x;
        double span_y = max_y - min_y;
        double span = fmax(span_x, span_y);
        if (span <= 1e-12) span = 1.0;

        double norm_x = 0.1 + 0.8 * (p.x - min_x) / span;
        double norm_y = 0.1 + 0.8 * (p.y - min_y) / span;

        uint32_t qx = (uint32_t)fmin(fmax(norm_x * 65535.0, 0.0), 65535.0);
        uint32_t qy = (uint32_t)fmin(fmax(norm_y * 65535.0, 0.0), 65535.0);

        cell_keys[idx] = morton2D(qx, qy);
        out_points[idx] = PointWithIdx(Point2D(norm_x, norm_y), idx);
    } else {
        cell_keys[idx] = 0xFFFFFFFFU;
        out_points[idx] = PointWithIdx(Point2D(0.0, 0.0), -1);
    }
}

// -----------------------------------------------------------------------------
// Bitonic Sort with Shared Memory & Conflict Minimization
// -----------------------------------------------------------------------------

__global__ void bitonic_sort_shared_kernel(
    uint32_t* d_keys,
    PointWithIdx* d_vals,
    int n,
    int padded_n
) {
    __shared__ uint32_t s_keys[1024];
    __shared__ PointWithIdx s_vals[1024];

    int tid = threadIdx.x;
    int idx = blockIdx.x * 1024 + tid;

    if (idx < padded_n) {
        s_keys[tid] = d_keys[idx];
        s_vals[tid] = d_vals[idx];
    } else {
        s_keys[tid] = 0xFFFFFFFFU;
        s_vals[tid] = PointWithIdx(Point2D(0.0, 0.0), -1);
    }
    __syncthreads();

    // Sort 1024 elements in shared memory
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (s_keys[tid] > s_keys[ixj]) {
                        uint32_t tk = s_keys[tid]; s_keys[tid] = s_keys[ixj]; s_keys[ixj] = tk;
                        PointWithIdx tv = s_vals[tid]; s_vals[tid] = s_vals[ixj]; s_vals[ixj] = tv;
                    }
                } else {
                    if (s_keys[tid] < s_keys[ixj]) {
                        uint32_t tk = s_keys[tid]; s_keys[tid] = s_keys[ixj]; s_keys[ixj] = tk;
                        PointWithIdx tv = s_vals[tid]; s_vals[tid] = s_vals[ixj]; s_vals[ixj] = tv;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (idx < padded_n) {
        d_keys[idx] = s_keys[tid];
        d_vals[idx] = s_vals[tid];
    }
}

__global__ void bitonic_merge_global_kernel(
    uint32_t* d_keys,
    PointWithIdx* d_vals,
    int j,
    int k,
    int padded_n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded_n) return;

    int ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0) {
            if (d_keys[i] > d_keys[ixj]) {
                uint32_t tk = d_keys[i]; d_keys[i] = d_keys[ixj]; d_keys[ixj] = tk;
                PointWithIdx tv = d_vals[i]; d_vals[i] = d_vals[ixj]; d_vals[ixj] = tv;
            }
        } else {
            if (d_keys[i] < d_keys[ixj]) {
                uint32_t tk = d_keys[i]; d_keys[i] = d_keys[ixj]; d_keys[ixj] = tk;
                PointWithIdx tv = d_vals[i]; d_vals[i] = d_vals[ixj]; d_vals[ixj] = tv;
            }
        }
    }
}

__global__ void bitonic_merge_shared_kernel(
    uint32_t* d_keys,
    PointWithIdx* d_vals,
    int k,
    int padded_n
) {
    __shared__ uint32_t s_keys[1024];
    __shared__ PointWithIdx s_vals[1024];

    int tid = threadIdx.x;
    int idx = blockIdx.x * 1024 + tid;
    if (idx >= padded_n) return;

    s_keys[tid] = d_keys[idx];
    s_vals[tid] = d_vals[idx];
    __syncthreads();

    for (int j = 512; j > 0; j >>= 1) {
        int ixj = tid ^ j;
        if (ixj > tid) {
            if ((idx & k) == 0) {
                if (s_keys[tid] > s_keys[ixj]) {
                    uint32_t tk = s_keys[tid]; s_keys[tid] = s_keys[ixj]; s_keys[ixj] = tk;
                    PointWithIdx tv = s_vals[tid]; s_vals[tid] = s_vals[ixj]; s_vals[ixj] = tv;
                }
            } else {
                if (s_keys[tid] < s_keys[ixj]) {
                    uint32_t tk = s_keys[tid]; s_keys[tid] = s_keys[ixj]; s_keys[ixj] = tk;
                    PointWithIdx tv = s_vals[tid]; s_vals[tid] = s_vals[ixj]; s_vals[ixj] = tv;
                }
            }
        }
        __syncthreads();
    }

    d_keys[idx] = s_keys[tid];
    d_vals[idx] = s_vals[tid];
}

__global__ void unpack_sorted_points(
    const PointWithIdx* d_sort_vals,
    Point2D* d_points,
    int* d_orig_indices,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_points[idx] = d_sort_vals[idx].pt;
        d_orig_indices[idx] = d_sort_vals[idx].orig_idx;
    }
}

// -----------------------------------------------------------------------------
// Delaunay Insertion & Triangulation Kernels
// -----------------------------------------------------------------------------

__global__ void init_super_triangle(
    Point2D* points,
    int3* triangles,
    int3* neighbors,
    int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Fixed super-triangle enclosing [0.1, 0.9]^2 domain
    points[n]     = Point2D(-100.0, -100.0);
    points[n + 1] = Point2D( 100.0, -100.0);
    points[n + 2] = Point2D(   0.0,  100.0);

    triangles[0] = make_int3(n, n + 1, n + 2);
    neighbors[0] = make_int3(-1, -1, -1);
}

__global__ void locate_and_vote(
    const Point2D* points,
    const int3* triangles,
    const int3* neighbors,
    int* point_tri,
    const int* inserted,
    int* tri_winner,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || inserted[idx]) return;

    Point2D p = points[idx];
    int t = point_tri[idx];

    for (int step = 0; step < 10000; ++step) {
        int3 v = triangles[t];
        int3 nbr = neighbors[t];

        // Duplicate vertex protection
        Point2D va = points[v.x], vb = points[v.y], vc = points[v.z];
        if ((fabs(p.x - va.x) < 1e-12 && fabs(p.y - va.y) < 1e-12) ||
            (fabs(p.x - vb.x) < 1e-12 && fabs(p.y - vb.y) < 1e-12) ||
            (fabs(p.x - vc.x) < 1e-12 && fabs(p.y - vc.y) < 1e-12)) {
            break;
        }

        double o0 = orient2d(vb, vc, p);
        double o1 = orient2d(vc, va, p);
        double o2 = orient2d(va, vb, p);

        int next_t = -1;
        double min_o = -1e-12;

        if (o0 < min_o && nbr.x != -1) { min_o = o0; next_t = nbr.x; }
        if (o1 < min_o && nbr.y != -1) { min_o = o1; next_t = nbr.y; }
        if (o2 < min_o && nbr.z != -1) { min_o = o2; next_t = nbr.z; }

        if (next_t != -1) {
            t = next_t;
        } else {
            break;
        }
    }

    point_tri[idx] = t;
    atomicMin(&tri_winner[t], idx);
}

__global__ void split_triangles(
    int3* triangles,
    int3* neighbors,
    const int* tri_winner,
    int* inserted,
    int* split_base,
    int* num_triangles,
    int prev_num_triangles,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= prev_num_triangles) return;

    int p = tri_winner[idx];
    if (p >= n) return;

    inserted[p] = 1;
    int base = atomicAdd(num_triangles, 2);
    split_base[idx] = base;

    int3 v = triangles[idx];
    int3 nbr = neighbors[idx];

    int c0 = idx;
    int c1 = base;
    int c2 = base + 1;

    triangles[c0] = make_int3(p,   v.y, v.z);
    triangles[c1] = make_int3(v.z, v.x, p);
    triangles[c2] = make_int3(v.x, v.y, p);

    neighbors[c0] = make_int3(nbr.x, c1,    c2);
    neighbors[c1] = make_int3(c2,    c0,    nbr.y);
    neighbors[c2] = make_int3(c0,    c1,    nbr.z);
}

__global__ void repair_neighbors(
    const int3* triangles,
    int3* neighbors,
    const int* split_base,
    int current_num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= current_num_triangles) return;

    int3 v = triangles[t];
    int3 nbr = neighbors[t];

    nbr.x = resolve_split_neighbor(triangles, split_base, nbr.x, v.y, v.z);
    nbr.y = resolve_split_neighbor(triangles, split_base, nbr.y, v.z, v.x);
    nbr.z = resolve_split_neighbor(triangles, split_base, nbr.z, v.x, v.y);
    neighbors[t] = nbr;
}

// -----------------------------------------------------------------------------
// Conflict-Free Parallel Lawson Edge Flipping Kernels
// -----------------------------------------------------------------------------

__global__ void propose_flips(
    const Point2D* points,
    const int3* triangles,
    const int3* neighbors,
    int* tri_lock,
    int* flip_edge,
    int* flip_partner,
    int* flip_count,
    int num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int3 v_t = triangles[t];
    int3 nbr_t = neighbors[t];

    int nbrs[3]   = {nbr_t.x, nbr_t.y, nbr_t.z};
    int2 edges[3] = {make_int2(v_t.y, v_t.z), make_int2(v_t.z, v_t.x), make_int2(v_t.x, v_t.y)};
    int opps[3]   = {v_t.x, v_t.y, v_t.z};

    for (int i = 0; i < 3; ++i) {
        int n = nbrs[i];
        if (n <= t || n >= num_triangles) continue;

        int A = opps[i];
        int B = edges[i].x;
        int C = edges[i].y;

        int3 v_n = triangles[n];
        int D = get_opposite_vertex(v_n, B, C);

        // Circumcircle criterion and strict convexity of quadrilateral ABDC
        if (is_in_circle(points[A], points[B], points[C], points[D]) &&
            orient2d(points[A], points[B], points[D]) > 0.0 &&
            orient2d(points[A], points[D], points[C]) > 0.0) {
            if (atomicCAS(&tri_lock[t], -1, t) == -1) {
                if (atomicCAS(&tri_lock[n], -1, t) == -1) {
                    flip_edge[t] = i;
                    flip_partner[t] = n;
                    flip_partner[n] = t;
                    atomicAdd(flip_count, 1);
                    return; // Participate in at most one flip per round
                } else {
                    atomicExch(&tri_lock[t], -1);
                }
            }
        }
    }
}

__global__ void execute_flips(
    int3* triangles,
    int3* neighbors,
    const int* flip_edge,
    const int* flip_partner,
    int num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int i = flip_edge[t];
    if (i == -1) return;

    int n = flip_partner[t];

    int3 v_t = triangles[t];
    int3 nbr_t = neighbors[t];
    int3 v_n = triangles[n];
    int3 nbr_n = neighbors[n];

    int A, B, C;
    if (i == 0)      { A = v_t.x; B = v_t.y; C = v_t.z; }
    else if (i == 1) { A = v_t.y; B = v_t.z; C = v_t.x; }
    else             { A = v_t.z; B = v_t.x; C = v_t.y; }

    int D = get_opposite_vertex(v_n, B, C);

    int n_BD = get_edge_neighbor(v_n, nbr_n, B, D);
    int n_DC = get_edge_neighbor(v_n, nbr_n, D, C);
    int n_AB = get_edge_neighbor(v_t, nbr_t, A, B);
    int n_CA = get_edge_neighbor(v_t, nbr_t, C, A);

    triangles[t] = make_int3(A, B, D);
    triangles[n] = make_int3(A, D, C);

    neighbors[t] = make_int3(n_BD, n, n_AB);
    neighbors[n] = make_int3(n_DC, n_CA, t);
}

__global__ void repair_flipped_neighbors(
    const int3* triangles,
    int3* neighbors,
    const int* flip_partner,
    int num_triangles
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_triangles) return;

    int3 v = triangles[t];
    int3 nbr = neighbors[t];

    nbr.x = resolve_flipped_neighbor(triangles, flip_partner, nbr.x, v.y, v.z);
    nbr.y = resolve_flipped_neighbor(triangles, flip_partner, nbr.y, v.z, v.x);
    nbr.z = resolve_flipped_neighbor(triangles, flip_partner, nbr.z, v.x, v.y);

    neighbors[t] = nbr;
}

// -----------------------------------------------------------------------------
// Stream Compaction / Output Filtering Kernel using Shared Memory
// -----------------------------------------------------------------------------

__global__ void filter_triangles_kernel(
    const int3* triangles,
    const int* orig_indices,
    int3* valid_triangles,
    int* valid_count,
    int num_triangles,
    int n
) {
    __shared__ int s_count;
    __shared__ int s_offset;

    int tid = threadIdx.x;
    if (tid == 0) s_count = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_valid = false;
    int3 mapped_tri = make_int3(-1, -1, -1);
    int my_loc = -1;

    if (idx < num_triangles) {
        int3 t = triangles[idx];
        if (t.x < n && t.y < n && t.z < n) {
            is_valid = true;
            mapped_tri = make_int3(
                orig_indices[t.x],
                orig_indices[t.y],
                orig_indices[t.z]
            );
            my_loc = atomicAdd(&s_count, 1);
        }
    }
    __syncthreads();

    if (tid == 0 && s_count > 0) {
        s_offset = atomicAdd(valid_count, s_count);
    }
    __syncthreads();

    if (is_valid) {
        valid_triangles[s_offset + my_loc] = mapped_tri;
    }
}

// -----------------------------------------------------------------------------
// Host Driver Implementation
// -----------------------------------------------------------------------------

bool run_delaunay_triangulation(
    const std::vector<Point2D>& h_in_points,
    std::vector<int3>& h_out_triangles,
    DelaunayStats& stats
) {
    int N = static_cast<int>(h_in_points.size());
    if (N < 3) {
        std::cerr << "Error: At least 3 points required for triangulation." << std::endl;
        return false;
    }

    stats.num_points = N;
    int max_triangles = 2 * N + 1;

    // Find next power of 2 for bitonic sort
    int padded_N = 1024;
    while (padded_N < N) padded_N <<= 1;

    const int threads = 256;
    int blocks_N = (N + threads - 1) / threads;
    int blocks_padded = (padded_N + threads - 1) / threads;

    // Create CUDA streams and events
    cudaStream_t compute_stream, reset_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&reset_stream));

    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_step, stop_step;
    cudaEvent_t reset_event, flip_reset_event;

    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_step));
    CUDA_CHECK(cudaEventCreate(&stop_step));
    CUDA_CHECK(cudaEventCreate(&reset_event));
    CUDA_CHECK(cudaEventCreate(&flip_reset_event));

    CUDA_CHECK(cudaEventRecord(start_total, compute_stream));

    // 1. Host Pinned Memory Allocation
    Point2D* h_pinned_points = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_points, N * sizeof(Point2D)));
    std::copy(h_in_points.begin(), h_in_points.end(), h_pinned_points);

    // 2. Device Memory Allocations
    Point2D* d_in_points = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in_points, N * sizeof(Point2D)));

    double *d_min_x = nullptr, *d_max_x = nullptr;
    double *d_min_y = nullptr, *d_max_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_min_x, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_max_x, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_min_y, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_max_y, sizeof(double)));

    uint32_t* d_sort_keys = nullptr;
    PointWithIdx* d_sort_vals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sort_keys, padded_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sort_vals, padded_N * sizeof(PointWithIdx)));

    Point2D* d_points = nullptr;
    int* d_orig_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, (N + 3) * sizeof(Point2D)));
    CUDA_CHECK(cudaMalloc(&d_orig_indices, N * sizeof(int)));

    int3* d_triangles = nullptr;
    int3* d_neighbors = nullptr;
    CUDA_CHECK(cudaMalloc(&d_triangles, max_triangles * sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, max_triangles * sizeof(int3)));

    int* d_point_tri = nullptr;
    int* d_inserted = nullptr;
    int* d_tri_winner = nullptr;
    int* d_split_base = nullptr;
    int* d_num_triangles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_point_tri, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inserted, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tri_winner, max_triangles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_split_base, max_triangles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_triangles, sizeof(int)));

    int* d_tri_lock = nullptr;
    int* d_flip_edge = nullptr;
    int* d_flip_partner = nullptr;
    int* d_flip_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tri_lock, max_triangles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flip_edge, max_triangles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flip_partner, max_triangles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flip_count, sizeof(int)));

    int3* d_valid_triangles = nullptr;
    int* d_valid_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_valid_triangles, max_triangles * sizeof(int3)));
    CUDA_CHECK(cudaMalloc(&d_valid_count, sizeof(int)));

    // -------------------------------------------------------------------------
    // Step 1: Copy points from Host RAM to Device Memory
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start_step, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in_points, h_pinned_points, N * sizeof(Point2D),
                               cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaEventRecord(stop_step, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_step));
    CUDA_CHECK(cudaEventElapsedTime(&stats.time_htod_ms, start_step, stop_step));

    // -------------------------------------------------------------------------
    // Step 2: Compute Bounding Box on GPU with Shared Memory Reduction
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start_step, compute_stream));

    double h_init_min = 1e30, h_init_max = -1e30;
    CUDA_CHECK(cudaMemcpyAsync(d_min_x, &h_init_min, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_max_x, &h_init_max, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_min_y, &h_init_min, sizeof(double), cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_max_y, &h_init_max, sizeof(double), cudaMemcpyHostToDevice, compute_stream));

    int bounds_blocks = std::min(64, blocks_N);
    compute_bounds_kernel<<<bounds_blocks, 256, 0, compute_stream>>>(
        d_in_points, d_min_x, d_max_x, d_min_y, d_max_y, N
    );

    double h_min_x, h_max_x, h_min_y, h_max_y;
    CUDA_CHECK(cudaMemcpyAsync(&h_min_x, d_min_x, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_max_x, d_max_x, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_min_y, d_min_y, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_max_y, d_max_y, sizeof(double), cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // -------------------------------------------------------------------------
    // Step 3: Morton Key Generation & GPU Bitonic Sort with Shared Memory
    // -------------------------------------------------------------------------
    normalize_and_morton_keys<<<blocks_padded, threads, 0, compute_stream>>>(
        d_in_points, d_sort_vals, d_sort_keys,
        h_min_x, h_max_x, h_min_y, h_max_y,
        N, padded_N
    );

    int bitonic_blocks = padded_N / 1024;
    bitonic_sort_shared_kernel<<<bitonic_blocks, 1024, 0, compute_stream>>>(
        d_sort_keys, d_sort_vals, N, padded_N
    );

    for (int k = 2048; k <= padded_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j >= 1024) {
                int merge_blocks = (padded_N + threads - 1) / threads;
                bitonic_merge_global_kernel<<<merge_blocks, threads, 0, compute_stream>>>(
                    d_sort_keys, d_sort_vals, j, k, padded_N
                );
            } else if (j == 512) {
                bitonic_merge_shared_kernel<<<bitonic_blocks, 1024, 0, compute_stream>>>(
                    d_sort_keys, d_sort_vals, k, padded_N
                );
                break;
            }
        }
    }

    unpack_sorted_points<<<blocks_N, threads, 0, compute_stream>>>(
        d_sort_vals, d_points, d_orig_indices, N
    );

    CUDA_CHECK(cudaEventRecord(stop_step, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_step));
    CUDA_CHECK(cudaEventElapsedTime(&stats.time_sort_ms, start_step, stop_step));

    // -------------------------------------------------------------------------
    // Step 4: Super-Triangle Initialization
    // -------------------------------------------------------------------------
    init_super_triangle<<<1, 1, 0, compute_stream>>>(
        d_points, d_triangles, d_neighbors, N
    );

    int init_triangles = 1;
    CUDA_CHECK(cudaMemcpyAsync(d_num_triangles, &init_triangles, sizeof(int),
                               cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_point_tri, 0, N * sizeof(int), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_inserted, 0, N * sizeof(int), compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_split_base, -1, max_triangles * sizeof(int), compute_stream));

    // -------------------------------------------------------------------------
    // Step 5: Iterative Incremental Point Insertion Loop
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start_step, compute_stream));

    int current_triangles = 1;
    int insertion_rounds = 0;

    while (current_triangles < max_triangles) {
        int prev_triangles = current_triangles;
        int t_blocks = (prev_triangles + threads - 1) / threads;

        // Multi-stream reset triggered using events
        fill_kernel<<<t_blocks, threads, 0, reset_stream>>>(d_tri_winner, INT_MAX, prev_triangles);
        CUDA_CHECK(cudaMemsetAsync(d_split_base, -1, prev_triangles * sizeof(int), reset_stream));
        CUDA_CHECK(cudaEventRecord(reset_event, reset_stream));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, reset_event, 0));

        // Locate containing triangle and vote
        locate_and_vote<<<blocks_N, threads, 0, compute_stream>>>(
            d_points, d_triangles, d_neighbors,
            d_point_tri, d_inserted, d_tri_winner,
            N
        );

        // Split winning triangles 1-to-3
        split_triangles<<<t_blocks, threads, 0, compute_stream>>>(
            d_triangles, d_neighbors, d_tri_winner,
            d_inserted, d_split_base, d_num_triangles,
            prev_triangles, N
        );

        int h_num_triangles;
        CUDA_CHECK(cudaMemcpyAsync(&h_num_triangles, d_num_triangles, sizeof(int),
                                   cudaMemcpyDeviceToHost, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        current_triangles = h_num_triangles;

        int all_blocks = (current_triangles + threads - 1) / threads;
        repair_neighbors<<<all_blocks, threads, 0, compute_stream>>>(
            d_triangles, d_neighbors, d_split_base, current_triangles
        );

        insertion_rounds++;
        if (current_triangles == prev_triangles) break;
    }

    CUDA_CHECK(cudaEventRecord(stop_step, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_step));
    CUDA_CHECK(cudaEventElapsedTime(&stats.time_insertion_ms, start_step, stop_step));
    stats.insertion_rounds = insertion_rounds;

    // -------------------------------------------------------------------------
    // Step 6: Conflict-Free 2-Phase Parallel Lawson Edge Flipping Loop
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start_step, compute_stream));

    int flip_rounds = 0;
    int total_flips = 0;
    int t_all_blocks = (current_triangles + threads - 1) / threads;

    while (true) {
        // Multi-stream reset of lock and flip buffers
        CUDA_CHECK(cudaMemsetAsync(d_tri_lock, -1, current_triangles * sizeof(int), reset_stream));
        CUDA_CHECK(cudaMemsetAsync(d_flip_edge, -1, current_triangles * sizeof(int), reset_stream));
        CUDA_CHECK(cudaMemsetAsync(d_flip_partner, -1, current_triangles * sizeof(int), reset_stream));
        CUDA_CHECK(cudaMemsetAsync(d_flip_count, 0, sizeof(int), reset_stream));
        CUDA_CHECK(cudaEventRecord(flip_reset_event, reset_stream));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, flip_reset_event, 0));

        propose_flips<<<t_all_blocks, threads, 0, compute_stream>>>(
            d_points, d_triangles, d_neighbors,
            d_tri_lock, d_flip_edge, d_flip_partner, d_flip_count,
            current_triangles
        );

        int h_flip_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_flip_count, d_flip_count, sizeof(int),
                                   cudaMemcpyDeviceToHost, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        if (h_flip_count == 0 || flip_rounds > MAX_FLIP_ROUNDS) break;

        execute_flips<<<t_all_blocks, threads, 0, compute_stream>>>(
            d_triangles, d_neighbors, d_flip_edge, d_flip_partner,
            current_triangles
        );

        repair_flipped_neighbors<<<t_all_blocks, threads, 0, compute_stream>>>(
            d_triangles, d_neighbors, d_flip_partner,
            current_triangles
        );

        total_flips += h_flip_count;
        flip_rounds++;
    }

    CUDA_CHECK(cudaEventRecord(stop_step, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_step));
    CUDA_CHECK(cudaEventElapsedTime(&stats.time_flip_ms, start_step, stop_step));
    stats.flip_rounds = flip_rounds;
    stats.total_flips = total_flips;

    // -------------------------------------------------------------------------
    // Step 7: Filter Valid Delaunay Triangles (Exclude Super-Triangle Vertices)
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start_step, compute_stream));

    CUDA_CHECK(cudaMemsetAsync(d_valid_count, 0, sizeof(int), compute_stream));
    filter_triangles_kernel<<<t_all_blocks, threads, 0, compute_stream>>>(
        d_triangles, d_orig_indices, d_valid_triangles, d_valid_count,
        current_triangles, N
    );

    int h_valid_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_valid_count, d_valid_count, sizeof(int),
                               cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    CUDA_CHECK(cudaEventRecord(stop_step, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_step));
    CUDA_CHECK(cudaEventElapsedTime(&stats.time_filter_ms, start_step, stop_step));
    stats.num_triangles = h_valid_count;

    // -------------------------------------------------------------------------
    // Step 8: Copy Output Triangles to Host Memory
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start_step, compute_stream));

    int3* h_pinned_triangles = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_triangles, h_valid_count * sizeof(int3)));

    CUDA_CHECK(cudaMemcpyAsync(h_pinned_triangles, d_valid_triangles,
                               h_valid_count * sizeof(int3),
                               cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    h_out_triangles.resize(h_valid_count);
    std::copy(h_pinned_triangles, h_pinned_triangles + h_valid_count, h_out_triangles.begin());

    CUDA_CHECK(cudaEventRecord(stop_step, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_step));
    CUDA_CHECK(cudaEventElapsedTime(&stats.time_dtoh_ms, start_step, stop_step));

    CUDA_CHECK(cudaEventRecord(stop_total, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop_total));
    CUDA_CHECK(cudaEventElapsedTime(&stats.time_total_gpu_ms, start_total, stop_total));

    // -------------------------------------------------------------------------
    // Step 9: Cleanup Device and Host Memory
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaFreeHost(h_pinned_points));
    CUDA_CHECK(cudaFreeHost(h_pinned_triangles));

    CUDA_CHECK(cudaFree(d_in_points));
    CUDA_CHECK(cudaFree(d_min_x));
    CUDA_CHECK(cudaFree(d_max_x));
    CUDA_CHECK(cudaFree(d_min_y));
    CUDA_CHECK(cudaFree(d_max_y));
    CUDA_CHECK(cudaFree(d_sort_keys));
    CUDA_CHECK(cudaFree(d_sort_vals));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_orig_indices));
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_point_tri));
    CUDA_CHECK(cudaFree(d_inserted));
    CUDA_CHECK(cudaFree(d_tri_winner));
    CUDA_CHECK(cudaFree(d_split_base));
    CUDA_CHECK(cudaFree(d_num_triangles));
    CUDA_CHECK(cudaFree(d_tri_lock));
    CUDA_CHECK(cudaFree(d_flip_edge));
    CUDA_CHECK(cudaFree(d_flip_partner));
    CUDA_CHECK(cudaFree(d_flip_count));
    CUDA_CHECK(cudaFree(d_valid_triangles));
    CUDA_CHECK(cudaFree(d_valid_count));

    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(reset_stream));
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_step));
    CUDA_CHECK(cudaEventDestroy(stop_step));
    CUDA_CHECK(cudaEventDestroy(reset_event));
    CUDA_CHECK(cudaEventDestroy(flip_reset_event));

    return true;
}
