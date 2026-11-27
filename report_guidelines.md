Role & Audience Constraints:
Write a clear, concise technical summary of this CUDA Delaunay Triangulation repository. 

Audience Profile:
- CUDA Knowledge: Understands CUDA basics (threads, blocks, grids, shared memory, streams, host vs. device transfers).
- Domain Knowledge: ZERO background in computational geometry or Delaunay Triangulation. They do NOT know what concepts like "point binning," "half-edges," "circumcircles," "Lawson flips," or "local mesh generation" mean.

Core Writing Rules:
1. Concept First, Term Second: Whenever you introduce a computational geometry step or data structure, explain *what it physically/visually does* in plain English BEFORE using its technical term or showing CUDA code.
2. Keep it Lean: Write in direct, concise technical prose. Avoid dense academic walls of text, unnecessary jargon, and generic filler.
3. Code Context: When showing pseudocode or CUDA memory access, clearly comment what problem that specific block of code solves in real-world terms.

Document Layout (Markdown):

1. Problem & Core Intuition
   - What problem does Delaunay Triangulation solve? Explain the "Empty Circumcircle" rule using a simple visual analogy (e.g., drawing a circle through 3 points that contains no other points) before giving the definition.
   - Where is this used? Give 2 concise real-world examples.
   - Data Formats: Define input (raw 2D coordinates) and output (triangles/edges) using simple C/CUDA `struct` definitions.

2. How the Algorithm Works (Step-by-Step)
   - High-Level Strategy: Explain how this specific repository breaks the 2D plane into smaller parallel tasks across GPU blocks. (Explain the partitioning visual step first before naming any binning or grid strategy).
   - Execution Pipeline: Detail the sequence of steps from loading points on the host to outputting the final triangles on the device.

3. Kernel Breakdown & Pseudocode
   - Host Setup: Pseudocode for host-side stream management, memory allocations, and kernel launches.
   - Device Kernels: Pseudocode for the main CUDA kernels. Annotate threads/blocks to explain who does what (e.g., "Thread X checks if Point Y falls inside Circle Z").

4. Shared Memory Usage
   - Purpose: What data is loaded into shared memory by thread blocks, and why is fast on-chip access needed for that specific data?
   - Access & Efficiency: Explain in simple terms how threads access this shared memory to avoid conflicts or redundant global memory reads.

5. Streams & Concurrency
   - Streaming Strategy: Explain how CUDA streams are organized (e.g., 2-way, 3-way, or N-way overlap).
   - Overlap: How memory transfers (`cudaMemcpyAsync`) and kernel execution run concurrently. Include a quick ASCII diagram showing stream execution over time.

6. Time Complexity & Performance
   - Complexity: State the time and space complexity ($O(N)$ vs $O(N^2)$) for best, average, and worst cases.
   - Main Bottleneck: Briefly state whether the algorithm spends most of its time computing math, waiting on memory, or handling thread divergence.

7. References & Lineage
   - Cite the underlying paper, foundational algorithm, or open-source projects this implementation is based on.
