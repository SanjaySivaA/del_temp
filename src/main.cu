#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include "geometry_types.h"
#include "delaunay.cuh"
#include "io.h"

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options] <input_file> <output_file>\n"
              << "Options:\n"
              << "  -i, --input <file>     Input 2D points file (PBBS sequencePoint2d format or coordinates)\n"
              << "  -o, --output <file>    Output triangles file (PBBS triangles format)\n"
              << "  -b, --benchmark        Display detailed GPU timing and execution metrics\n"
              << "  -s, --simple           Also export points.txt and triangles.txt for visualization\n"
              << "  -h, --help             Display this help message\n\n"
              << "Workflow:\n"
              << "  " << prog_name << " points.in triangles.out\n"
              << "  " << prog_name << " -i input.2d -o output.tri --benchmark\n";
}

int main(int argc, char** argv) {
    std::string input_file = "";
    std::string output_file = "";
    bool show_benchmark = true;
    bool export_simple = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) input_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_file = argv[++i];
        } else if (arg == "-b" || arg == "--benchmark") {
            show_benchmark = true;
        } else if (arg == "-s" || arg == "--simple") {
            export_simple = true;
        } else if (arg[0] != '-') {
            if (input_file.empty()) {
                input_file = arg;
            } else if (output_file.empty()) {
                output_file = arg;
            }
        }
    }

    if (input_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Step 1: Read input points
    std::vector<Point2D> points;
    if (!read_pbbs_points(input_file, points)) {
        std::cerr << "Failed to read input points from: " << input_file << std::endl;
        return 1;
    }

    std::cout << "Loaded " << points.size() << " 2D points from " << input_file << std::endl;

    // Step 2: Run Delaunay Triangulation on GPU
    std::vector<int3> triangles;
    DelaunayStats stats;
    if (!run_delaunay_triangulation(points, triangles, stats)) {
        std::cerr << "Delaunay triangulation execution failed." << std::endl;
        return 1;
    }

    // Step 3: Write PBBS Triangles Output
    if (!output_file.empty()) {
        if (!write_pbbs_triangles(output_file, points, triangles)) {
            std::cerr << "Failed to write PBBS triangles to: " << output_file << std::endl;
            return 1;
        }
        std::cout << "Wrote " << triangles.size() << " Delaunay triangles to " << output_file << std::endl;
    }

    // Optional: Export simple formats for plotting
    if (export_simple) {
        write_simple_points("points.txt", points);
        write_simple_triangles("triangles.txt", triangles);
        std::cout << "Exported points.txt and triangles.txt for visualization." << std::endl;
    }

    // Display Benchmark Statistics
    if (show_benchmark) {
        std::cout << "\n====================================================\n";
        std::cout << "      GPU Delaunay Triangulation Benchmark Results   \n";
        std::cout << "====================================================\n";
        std::cout << " Points Inserted         : " << stats.num_points << "\n";
        std::cout << " Final Triangles         : " << stats.num_triangles << "\n";
        std::cout << " Insertion Parallel Waves: " << stats.insertion_rounds << " rounds\n";
        std::cout << " Lawson Edge Flips       : " << stats.total_flips 
                  << " (" << stats.flip_rounds << " rounds)\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << " Host-to-Device (HtoD)   : " << stats.time_htod_ms << " ms\n";
        std::cout << " Bounds & Morton Sort   : " << stats.time_sort_ms << " ms\n";
        std::cout << " Point Insertion Phase   : " << stats.time_insertion_ms << " ms\n";
        std::cout << " Lawson Edge Flip Phase  : " << stats.time_flip_ms << " ms\n";
        std::cout << " Stream Compaction Filter: " << stats.time_filter_ms << " ms\n";
        std::cout << " Device-to-Host (DtoH)   : " << stats.time_dtoh_ms << " ms\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << " Total GPU Execution Time: " << stats.time_total_gpu_ms << " ms\n";
        std::cout << "====================================================\n";
    }

    return 0;
}
