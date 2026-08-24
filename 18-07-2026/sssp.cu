#include <iostream>
#include <vector>
#include <cfloat>
#include <cuda_runtime.h>

struct Edge {
    int dest;
    float weight;
};

struct TreeInfo {
    float dist;
    int parent;
};

__global__ void bellman_ford_relax_kernel(
    int nVertices,
    const int *row_offset,
    const Edge *out_edges,
    TreeInfo *tree_info,
    bool *d_relaxed
) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src >= nVertices) return;

    float src_dist = tree_info[src].dist;

    if (src_dist >= FLT_MAX) return;

    int edge_start = row_offset[src];
    int edge_end   = row_offset[src + 1];

    for (int e = edge_start; e < edge_end; e++) {
        int dest = out_edges[e].dest;
        float weight = out_edges[e].weight;
        float new_dist = src_dist + weight;

        float old_dist = atomicMin(&tree_info[dest].dist, new_dist);

        if (new_dist < old_dist) {
            tree_info[dest].parent = src;
            *d_relaxed = true;
        }
    }
}

int main() {
    /*
     * A graph with 5 vertices:
     *
     * 0 ---(6 )--> 1
     * 0 ---(7 )--> 2
     * 1 ---(8 )--> 2
     * 1 ---(5 )--> 3
     * 1 ---(-4)--> 4
     * 2 ---(-3)--> 3
     * 2 ---(9 )--> 4
     * 3 ---(-2)--> 1
     * 4 ---(2 )--> 0
     * 4 ---(7 )--> 3
     */

    const int nVertices = 5;
    const int nEdges = 10;
    const int source_vertex = 0;

    std::vector<int> h_row_offset = {0, 2, 5, 7, 8, 10};
    std::vector<Edge> h_out_edges = {
        {1,  6.0f}, {2,  7.0f},
        {2,  8.0f}, {3,  5.0f}, {4, -4.0f},
        {3, -3.0f}, {4,  9.0f},
        {1, -2.0f},
        {0,  2.0f}, {3,  7.0f}
    };

    std::vector<TreeInfo> h_tree_info(nVertices);
    for (int i = 0; i < nVertices; ++i) {
        h_tree_info[i].dist = FLT_MAX;
        h_tree_info[i].parent = -1;
    }
    h_tree_info[source_vertex].dist = 0.0f;

    int *d_row_offset;
    Edge *d_out_edges;
    TreeInfo *d_tree_info;
    bool *d_relaxed;

    cudaMalloc(&d_row_offset, (nVertices + 1) * sizeof(int));
    cudaMalloc(&d_out_edges, nEdges * sizeof(Edge))
    cudaMalloc(&d_tree_info, nVertices * sizeof(TreeInfo));
    cudaMalloc(&d_relaxed, sizeof(bool));

    cudaMemcpy(d_row_offset, h_row_offset.data(), (nVertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_edges, h_out_edges.data(), nEdges * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_info, h_tree_info.data(), nVertices * sizeof(TreeInfo), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (nVertices + blockSize - 1) / blockSize;

    bool h_relaxed = true;
    int iteration = 0;

    while (h_relaxed && iteration < nVertices - 1) {
        h_relaxed = false;
        cudaMemcpy(d_relaxed, &h_relaxed, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_relax_kernel<<<numBlocks, blockSize>>>(
            nVertices,
            d_row_offset,
            d_out_edges,
            d_tree_info,
            d_relaxed
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_relaxed, d_relaxed, sizeof(bool), cudaMemcpyDeviceToHost);
        iteration++;
    }

    bool has_negative_cycle = false;
    if (h_relaxed && iteration == nVertices - 1) {
        h_relaxed = false;
        cudaMemcpy(d_relaxed, &h_relaxed, sizeof(bool), cudaMemcpyHostToDevice);

        bellman_ford_relax_kernel<<<numBlocks, blockSize>>>(
            nVertices,
            d_row_offset,
            d_out_edges,
            d_tree_info,
            d_relaxed
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_relaxed, d_relaxed, sizeof(bool), cudaMemcpyDeviceToHost);
        if (h_relaxed) {
            has_negative_cycle = true;
        }
    }

    cudaMemcpy(h_tree_info.data(), d_tree_info, nVertices * sizeof(TreeInfo), cudaMemcpyDeviceToHost);

    if (has_negative_cycle) {
        std::cout << "Warning: Graph contains a negative-weight cycle reachable from source." << std::endl;
    } else {
        std::cout << "SSSP: source vertex " << source_vertex << " (Iterations: " << iteration << "):\n";
        std::cout << "Vertex\tDistance\tParent\n";
        std::cout << "--------------------------------\n";
        for (int i = 0; i < nVertices; ++i) {
            std::cout << i << "\t";
            if (h_tree_info[i].dist >= FLT_MAX) {
                std::cout << "INF\t\t";
            } else {
                std::cout << h_tree_info[i].dist << "\t\t";
            }
            std::cout << h_tree_info[i].parent << "\n";
        }
    }

    cudaFree(d_row_offset);
    cudaFree(d_out_edges);
    cudaFree(d_tree_info);
    cudaFree(d_relaxed);

    return 0;
}
