#include "hnswlib_dir/search.hpp"

/*
Old code with single-thread HNSWLib
*/

std::pair<std::vector<std::vector<hnswlib::labeltype>>, std::vector<std::vector<float>>> search(hnswlib::HierarchicalNSW<float> *index, const std::vector<std::vector<float>> &query_data)
{
    // Search parameters
    int k = Config::Search::K;

    // Validate query data
    if (query_data.empty())
    {
        throw std::runtime_error("Query data is empty");
    }

    // Perform searches
    std::vector<std::vector<hnswlib::labeltype>> results;
    std::vector<std::vector<float>> distances;

    for (size_t i = 0; i < query_data.size(); i++)
    {
        auto result = index->searchKnnCloserFirst(query_data[i].data(), k);

        std::vector<hnswlib::labeltype> query_results;
        std::vector<float> query_distances;

        // Extract results
        for (const auto &neighbor : result)
        {
            query_distances.push_back(neighbor.first); // distance
            query_results.push_back(neighbor.second);  // label/id
        }

        results.push_back(query_results);
        distances.push_back(query_distances);
    }

    return std::make_pair(results, distances);
}