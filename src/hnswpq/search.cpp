#include "hnswpq/search.hpp"

/*
Multi-threaded FAISS IndexHNSWPQ search using OpenMP
*/
std::pair<std::vector<std::vector<faiss::idx_t>>, std::vector<std::vector<float>>> faiss_search(
    faiss::IndexHNSWPQ *index,
    const std::vector<std::vector<float>> &query_data,
    int k,
    int ef)
{
    // Set search parameters
    index->hnsw.efSearch = ef;

    // Validate query data
    if (query_data.empty())
    {
        throw std::runtime_error("Query data is empty");
    }

    int num_queries = query_data.size();
    int dim = query_data[0].size();

    // Pre-allocate result vectors
    std::vector<std::vector<faiss::idx_t>> results(num_queries);
    std::vector<std::vector<float>> distances(num_queries);

    // Flatten query data for FAISS batch search
    std::vector<float> queries_flat(num_queries * dim);
    for (int i = 0; i < num_queries; ++i) {
        std::copy(query_data[i].begin(), query_data[i].end(),
                 queries_flat.begin() + i * dim);
    }

    // FAISS batch search (automatically parallelized)
    std::vector<float> batch_distances(num_queries * k);
    std::vector<faiss::idx_t> batch_labels(num_queries * k);

    index->search(num_queries, queries_flat.data(), k,
                 batch_distances.data(), batch_labels.data());

    // Convert batch results back to vector of vectors format
    for (int i = 0; i < num_queries; ++i) {
        std::vector<faiss::idx_t> query_results(k);
        std::vector<float> query_distances(k);

        for (int j = 0; j < k; ++j) {
            query_results[j] = batch_labels[i * k + j];
            query_distances[j] = batch_distances[i * k + j];
        }

        results[i] = std::move(query_results);
        distances[i] = std::move(query_distances);
    }

    return std::make_pair(results, distances);
}

