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

struct graph {
    int *edges;
    int *offsets;
    size_t nVertices;
    size_t nEdges;
};

struct edge {
    int src;
    int dest;
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

    CHECK_CUDA( cudaMallocManaged(&g.edges, nEdges * sizeof(int)) );
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
        g.edges[insert_idx] = edge.dest;
    }

    return g;
}

__global__ void BFS(graph g, int *dist, bool *changed, int dist_frontier) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src >= g.nVertices) return;

    if (dist[src] != dist_frontier) return;

    int start_edge = g.offsets[src];
    int end_edge = g.offsets[src + 1];

    for (int neighIdx = start_edge; neighIdx < end_edge; neighIdx++) {
        int dest = g.edges[neighIdx];
        int new_dist = dist_frontier + 1;

        int old_dist = new_dist;

        if (new_dist < old_dist) {
            *changed = true;
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
        {0, 1},
        {0, 2},
        {1, 2},
        {1, 3},
        {2, 4},
        {4, 3},
        {3, 5},
        {4, 5}
    };

    graph g = assoc_to_csr(edges);

    int source_vertex = 0;

    // Setup `dist` array, which tracks the distance of each vertex from the
    // source vertex
    int *dist;
    CHECK_CUDA( cudaMallocManaged(&dist, g.nVertices * sizeof(int)) );

    for (int i = 0; i < g.nVertices; i++) {
        dist[i] = INT_MAX;
    }
    dist[source_vertex] = 0;

    bool *changed;
    CHECK_CUDA( cudaMallocManaged(&changed, sizeof(bool)) );

    int threadsPerBlock = 256;
    int blocksPerGrid = (g.nVertices + threadsPerBlock - 1) / threadsPerBlock;

    int dist_frontier = 0;

    while(true) {
        *changed = false;

        BFS<<<blocksPerGrid, threadsPerBlock>>>(g, dist, changed, dist_frontier);
        CHECK_CUDA( cudaGetLastError() );
        CHECK_CUDA( cudaDeviceSynchronize() );

        if (!*changed) {
            std::cout << "Converged after " << dist_frontier + 1 << " iterations.\n\n";
            break;
        }

        dist_frontier++;
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
