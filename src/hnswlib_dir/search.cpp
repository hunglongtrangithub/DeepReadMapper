#include "hnswlib_dir/search.hpp"

/*
Multi-threaded HNSWLib search using OpenMP
*/

std::pair<std::vector<std::vector<hnswlib::labeltype>>, std::vector<std::vector<float>>> search(hnswlib::HierarchicalNSW<float> *index, const std::vector<std::vector<float>> &query_data)
{
    // Config multi-threaded search parameters
    omp_set_num_threads(Config::Search::NUM_THREADS);

    // Search parameters
    int k = Config::Search::K;

    // Validate query data
    if (query_data.empty())
    {
        throw std::runtime_error("Query data is empty");
    }

    // Pre-allocate result vectors
    std::vector<std::vector<hnswlib::labeltype>> results(query_data.size());
    std::vector<std::vector<float>> distances(query_data.size());

// Parallel search using OpenMP
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < query_data.size(); i++)
    {
        auto result = index->searchKnnCloserFirst(query_data[i].data(), k);

        // Pre-allocate vectors for this query
        std::vector<hnswlib::labeltype> query_results;
        std::vector<float> query_distances;
        query_results.reserve(result.size());
        query_distances.reserve(result.size());

        // Extract results
        for (const auto &neighbor : result)
        {
            query_distances.push_back(neighbor.first); // distance
            query_results.push_back(neighbor.second);  // label/id
        }

        // Store results
        results[i] = std::move(query_results);
        distances[i] = std::move(query_distances);
    }

    return std::make_pair(results, distances);
}