#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t const status = (call);                                    \
        if (cudaSuccess != status) {                                          \
            std::string error_msg = "CUDA Error: " +                          \
                                    std::string(cudaGetErrorString(status)) + \
                                    " at " + __FILE__ + ":" +                 \
                                    std::to_string(__LINE__);                 \
            throw std::runtime_error(error_msg);                              \
        }                                                                     \
    } while (0)

using weight_t = int;

struct cell {
    int dest;
    weight_t weight;
};

struct graph {
    cell *edges;
    int *offsets;
    size_t nVertices;
    size_t nEdges;
};

struct edge {
    int src;
    int dest;
    weight_t weight;
};

graph assoc_to_csr(const std::vector<edge>& edges) {
    int nEdges = edges.size();
    int nVertices = 0;

    // Find number of vertices
    for (const auto& edge : edges) {
        nVertices = std::max(nVertices, std::max(edge.dest, edge.src));
    }
    nVertices++;

    graph g;
    g.nEdges = nEdges;
    g.nVertices = nVertices;

    CHECK_CUDA( cudaMallocManaged(&g.edges, nEdges * sizeof(cell)) );
    CHECK_CUDA( cudaMallocManaged(&g.offsets, (nVertices + 1) * sizeof(int)) );

    for (int i = 0; i <= nVertices; i++) {
        g.offsets[i] = 0;
    }

    for (const auto& edge : edges) {
        g.offsets[edge.src + 1]++;
    }

    for (int i = 0; i < nVertices; i++) {
        g.offsets[i + 1] += g.offsets[i];
    }

    std::vector<int> current_offset(g.offsets, g.offsets + nVertices);

    for (const auto& edge : edges) {
        int insert_idx = current_offset[edge.src]++;
        g.edges[insert_idx] = { edge.dest, edge.weight };
    }

    return g;
}

__global__ void SSSP(graph g, weight_t *dist, bool *changed, const bool *updated_current, bool *updated_next) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src >= g.nVertices) return;

    if (!updated_current[src]) return;

    int start_edge = g.offsets[src];
    int end_edge = g.offsets[src + 1];

    for (int neighIdx = start_edge; neighIdx < end_edge; neighIdx++) {
        cell edge = g.edges[neighIdx];
        int dest = edge.dest;
        weight_t new_dist = dist[src] + edge.weight;

        weight_t old_dist = atomicMin(&dist[dest], new_dist);

        if (new_dist < old_dist) {
            *changed = true;

            updated_next[dest] = true;
        }
    }
}

void free_csr(graph g) {
    CHECK_CUDA( cudaFree(g.edges) );
    CHECK_CUDA( cudaFree(g.offsets) );
}

int main() {
    // Example graph
    std::vector<edge> edges = {
        {0, 1, 4},
        {0, 2, 2},
        {1, 2, 5},
        {1, 3, 10},
        {2, 4, 3},
        {4, 3, 4},
        {3, 5, 11},
        {4, 5, 2}
    };

    graph g = assoc_to_csr(edges);

    int source_vertex = 0;

    // Setup `dist` array, which tracks the distance of each vertex from the
    // source vertex
    weight_t *dist;
    CHECK_CUDA( cudaMallocManaged(&dist, g.nVertices * sizeof(weight_t)) );

    for (int i = 0; i < g.nVertices; i++) {
        dist[i] = INT_MAX;
    }
    dist[source_vertex] = 0;

    bool *updated_current;
    bool *updated_next;
    CHECK_CUDA( cudaMallocManaged(&updated_current, g.nVertices * sizeof(bool)) );
    CHECK_CUDA( cudaMallocManaged(&updated_next, g.nVertices * sizeof(bool)) );

    CHECK_CUDA( cudaMemset(updated_current, 0, g.nVertices * sizeof(bool)) );
    CHECK_CUDA( cudaMemset(updated_next, 0, g.nVertices * sizeof(bool)) );
    updated_current[source_vertex] = true;

    bool *changed;
    CHECK_CUDA( cudaMallocManaged(&changed, sizeof(bool)) );

    int threadsPerBlock = 256;
    int blocksPerGrid = (g.nVertices + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < g.nVertices - 1; i++) {
        *changed = false;

        SSSP<<<blocksPerGrid, threadsPerBlock>>>(g, dist, changed, updated_current, updated_next);
        CHECK_CUDA( cudaGetLastError() );
        CHECK_CUDA( cudaDeviceSynchronize() );

        if (!*changed) {
            std::cout << "Converged after " << i + 1 << " iterations.\n\n";
            break;
        }

        std::swap(updated_current, updated_next);

        // Clear the new updated_next array for the upcoming iteration
        CHECK_CUDA( cudaMemset(updated_next, 0, g.nVertices * sizeof(bool)) );
    }

    std::cout << "Shortest path distances from vertex " << source_vertex << ":\n";
    for (int i = 0; i < g.nVertices; i++) {
        if (dist[i] == INT_MAX) {
            std::cout << "Node " << i << ": INF\n";
        }
        else {
            std::cout << "Node " << i << ": " << dist[i] << "\n";
        }
    }

    cudaFree(dist);
    cudaFree(changed);
    free_csr(g);

    return 0;
}
