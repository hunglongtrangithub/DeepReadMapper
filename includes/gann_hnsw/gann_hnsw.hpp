#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <memory>
#include <thread>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <immintrin.h>                // For SIMD operations
#include <cereal/archives/binary.hpp> // For serialization
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

/// @brief GannHNSW class for creating a customized, high-performance HNSW based on GANN's paper architecture.
/// @details This class provides methods to build an HNSW index and perform advanced search operations.
class GannHNSW
{
public:
    // Type aliases for convenience
    using VertexId = uint32_t;
    using Distance = float;

    struct SearchResult
    {
        std::vector<VertexId> ids;
        std::vector<Distance> distances;
    };

    /**
     * @brief Constructor
     * @param dimension Dimensionality of the feature vectors
     * @param max_elements Maximum number of elements that can be stored (adopt from HNSWLib design)
     * @param M Graph degree parameter (connections per node)
     * @param ef_construction Size of the dynamic candidate list during construction (EFC)
     * @param ml Level generation factor (default 1/ln(2.0))
     */
    GannHNSW(
        size_t dimension,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        double ml = 1.0 / std::log(2.0))
        : dimension_(dimension),
          max_elements_(max_elements),
          M_(M),
          dmin_(M),     // dmin = M for NSW construction
          dmax_(M * 2), // dmax = 2*M to allow for bidirectional connections
          ef_construction_(ef_construction),
          ml_(ml),
          num_elements_(0),
          max_level_(16),
          level_generator_(std::random_device{}()),
          level_distribution_(0.0, 1.0)
    {
        data_.reserve(max_elements);
        entry_point_ = std::numeric_limits<VertexId>::max();
    }

    /**
     * @brief Build an HNSW index from input data.
     * @param input_data A 2D array of input vectors, where each vector is a 1D array of floats.
     * @param num_threads Number of threads to use for parallel construction (default: 0 for auto-detect)
     */
    void build(const std::vector<std::vector<float>> &input_data, size_t num_threads = 0);

    /**
     * @brief Search for k nearest neighbors using GANN's lazy update strategy
     * @param query_data 2D vector of query vectors
     * @param k Number of nearest neighbors to return
     * @param ef Search parameter (size of dynamic candidate list)
     * @param num_threads Number of threads for parallel search (default: hardware concurrency)
     * @return SearchResult containing neighbor IDs and distances
     */
    SearchResult search(const std::vector<std::vector<float>> &queryData, size_t k, size_t ef = 50, size_t num_threads = 0) const;

    /**
     * @brief Save the built index to file
     * @param index_file Path to save the index file
     */
    void save(const std::string &index_file) const;

    /**
     * @brief Load index from file
     * @param index_file Path to the index file
     * @return True if loaded successfully, false otherwise
     */
    bool load(const std::string &index_file);

    /**
     * @brief Get the number of elements in the index
     */
    size_t size() const { return num_elements_; }

    /**
     * @brief Get the dimensionality of feature vectors
     */
    size_t dimension() const { return dimension_; }

    /**
     * @brief A utility function to save/load the index using Cereal serialization
     * @tparam Archive Type of the archive (binary, JSON, etc.)
     * @param ar The archive to serialize to/from
     */
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::make_nvp("dimension", const_cast<size_t &>(dimension_)),
           cereal::make_nvp("max_elements", const_cast<size_t &>(max_elements_)),
           cereal::make_nvp("M", const_cast<size_t &>(M_)),
           cereal::make_nvp("dmin", const_cast<size_t &>(dmin_)),
           cereal::make_nvp("dmax", const_cast<size_t &>(dmax_)),
           cereal::make_nvp("ef_construction", const_cast<size_t &>(ef_construction_)),
           cereal::make_nvp("ml", const_cast<double &>(ml_)),
           cereal::make_nvp("data", data_),
           cereal::make_nvp("layers", layers_),
           cereal::make_nvp("entry_point", entry_point_),
           cereal::make_nvp("num_elements", num_elements_),
           cereal::make_nvp("max_level", max_level_));
    }

private:
    // Core GANN data structures
    struct Vertex
    {
        VertexId id;
        std::vector<VertexId> neighbors;   // Adjacency list
        bool explored = false;             // For lazy check
        Distance distance_to_query = 0.0f; // Cached distance

        // Cereal serialization
        template <class Archive>
        void serialize(Archive &ar)
        {
            ar(id, neighbors, explored, distance_to_query);
        }
    };

    struct Layer
    {
        std::vector<Vertex> vertices;
        size_t level;

        // Cereal serialization
        template <class Archive>
        void serialize(Archive &ar)
        {
            ar(vertices, level);
        }
    };

    // GANN search arrays
    struct SearchContext
    {
        std::vector<Vertex> N; // Top-k results + exploration candidates
        std::vector<Vertex> T; // Visiting vertices (neighbors)
        size_t ln;             // Length of N array
        size_t lt;             // Length of T array
    };

    // Index parameters
    const size_t dimension_;
    const size_t max_elements_;
    const size_t M_;               // Graph degree
    const size_t ef_construction_; // Construction parameter
    const double ml_;              // Level generation factor

    // GANN index params
    const size_t dmin_;
    const size_t dmax_;

    // Index data
    std::vector<std::vector<float>> data_; // ref vectors
    std::vector<Layer> layers_;            // Hierarchical layers
    VertexId entry_point_;                 // Global entry point
    size_t num_elements_;                  // Current number of elements
    size_t max_level_;                     // Maximum level in hierarchy

    // Random number generation for level selection
    mutable std::mt19937 level_generator_;
    mutable std::uniform_real_distribution<double> level_distribution_;

    // GANN index construction
    void buildLayer(size_t level, const std::vector<VertexId> &vertices, size_t num_threads);
    void buildLocalGraphsParallel(const std::vector<VertexId> &vertices, size_t level, size_t num_threads);
    void mergeLocalGraphs(size_t level);

    // GANN search implementation
    SearchResult searchLayer(const std::vector<float> &query, size_t k, size_t ef, size_t level, VertexId entry_point) const;
    SearchResult searchQuery(const std::vector<float> &query, size_t k, size_t ef) const;

    void computeDistanceParallel(const std::vector<float> &query, std::vector<Vertex> &candidates) const;

    // Utility functions
    Distance computeDistance(const std::vector<float> &a, const std::vector<float> &b) const;
    size_t getRandomLevel() const;
};
