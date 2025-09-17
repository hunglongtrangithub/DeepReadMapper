#include "utils.hpp"
#include "config.hpp"
#include "vectorize.hpp"
#include "parse_inputs.hpp"
#include "reranker.hpp"
#include "progressbar.h"
#include <memory>
#include <functional>
#include <type_traits>
#include <chrono>
#include <omp.h>

struct PositionInfo
{
    size_t position; // Actual position in the original sequence (FASTA)
    int RC;          // 0 for forward strand, 1 for reverse complement
};

/// @brief Decode position and strand information from a label. //! Only use for special HNSW index with encoded labels. Ignore if using default sequential labels.
/// @param label HNSWPQ sequential label
/// @param mapping Custom vector that map sequential label to original position
/// @return PositionInfo struct containing position and RC flag
PositionInfo position_decode(size_t label, const std::vector<size_t> &mapping);

/// @brief Reformat raw HNSW output (vector of pairs) into two separate vectors - templated to handle any neighbor type
/// @param neighbors Vector of vectors containing neighbor indices (size_t or long int)
/// @param distances Vector of vectors containing L2 distances from query to neighbors
/// @return Pair of vectors: 1st is labels, 2nd is distances
template <typename NeighborType>
std::pair<std::vector<size_t>, std::vector<float>> reformat_output(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    size_t k);

/// @brief Find full-length sequence from a reference genome string. This method is called "dynamic fetching" in which the candidate sequences are generated during search time.
/// @param ref_genome Reference genome as a single string
/// @param id Id of the sequence to find
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @return Sequence string, or empty string if not found
std::string find_sequence(const std::string &ref_genome, size_t id, size_t ref_len);

/// @brief Find full-length sequence from a vector of reference sequences. This method is called "static fetching" in which the candidate sequences are pre-extracted and stored in a vector.
/// @param ref_seqs Vector of reference sequences
/// @param id Id of the sequence to find
/// @return Sequence string
std::string find_sequence_static(const std::vector<std::string> &ref_seqs, size_t id);

/// @brief Wrapper to find multiple sequences asynchronously from dynamic reference genome
/// @param ref_genome Reference genome as a single string
/// @param ids Vector of ids to find
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @return Vector of sequences found, in the same order as ids
std::vector<std::string> find_sequences(const std::string &ref_genome, const std::vector<size_t> &ids, size_t ref_len, size_t stride = 1);

/// @brief Wrapper to find multiple sequences asynchronously from static reference sequences (Overload for static fetching)
/// @param ref_seqs Vector of reference sequences
/// @param ids Vector of ids to find
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @return Vector of sequences found, in the same order as ids
std::vector<std::string> find_sequences(const std::vector<std::string> &ref_seqs, const std::vector<size_t> &ids, size_t stride = 1);

/// @brief Helper function to convert neighbor types to size_t
template <typename NeighborType>
/// @param neighbors 2D vector of neighbor indices (any integral type)
/// @return 2D vector of neighbor indices as size_t
std::vector<std::vector<size_t>> convert_neighbors(const std::vector<std::vector<NeighborType>> &neighbors);

/// @brief Post-process with Smith-Waterman reranking using dynamic sequence fetching
/// @param neighbors 2D vector of neighbor indices from HNSW search (size_t or long int)
/// @param distances 2D vector of distances from HNSW search
/// @param ref_genome Reference genome as a single string
/// @param query_seqs Vector of query sequences
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @param k Number of final candidates to return per query
/// @param rerank_lim top-k candidates passed to reranker (default: 5)
/// @return Pair of vectors: 1st is sequences, 2nd is Smith-Waterman scores
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<int>> post_process_sw(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k, size_t rerank_lim = 5);

/// @brief Post-process with L2 distance reranking using dynamic sequence fetching
/// @param neighbors 2D vector of neighbor indices from HNSW search (size_t or long int)
/// @param distances 2D vector of distances from HNSW search
/// @param ref_genome Reference genome as a single string
/// @param query_seqs Vector of query sequences
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @param k Number of final candidates to return per query
/// @param query_embeddings Pre-computed embeddings for query sequences
/// @param vectorizer Vectorizer instance for computing candidate embeddings
/// @param rerank_lim top-k candidates passed to reranker (default: 5)
/// @return Pair of vectors: 1st is sequences, 2nd is L2 distances
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_dynamic(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer, size_t rerank_lim = 5);

/// @brief Post-process with L2 distance reranking using static sequence fetching
/// @param neighbors 2D vector of neighbor indices from HNSW search (size_t or long int)
/// @param distances 2D vector of distances from HNSW search
/// @param ref_seqs Vector of pre-extracted reference sequences
/// @param query_seqs Vector of query sequences
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @param k Number of final candidates to return per query
/// @param query_embeddings Pre-computed embeddings for query sequences
/// @param vectorizer Vectorizer instance for computing candidate embeddings
/// @param rerank_lim top-k candidates passed to reranker (default: 5)
/// @note During configuration for rerank_lim and k, ensure rerank_lim <= k / stride to guarantee enough candidates for dense index retrieval.
/// @return Pair of vectors: 1st is sequences, 2nd is L2 distances
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_static(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer, size_t rerank_lim = 5);