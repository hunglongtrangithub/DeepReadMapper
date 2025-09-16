#include "utils.hpp"
#include "config.hpp"
#include "vectorize.hpp"
#include "parse_inputs.hpp"
#include "reranker.hpp"
#include "progressbar.h"
#include <omp.h>
#include <memory>
#include <functional>
#include <type_traits>
#include <chrono>

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

/// @brief A template wrapper to handle the post-processing step after HNSW search with custom reranker
/// @param neighbors 2D vector of neighbor indices from HNSW search
/// @param distances 2D vector of distances from HNSW search
/// @param ref_genome Reference genome as a single string
/// @param queries Vector of query sequences (with PREFIX/POSTFIX)
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @param k Number of final candidates to return per query
/// @param rerank_func A callable function that takes in candidate sequences and a query, and returns top k sequences and their scores
/// @return Pair of vectors: 1st is sequences, 2nd is scores
template <typename QueryType, typename ScoreType>
std::pair<std::vector<std::string>, std::vector<ScoreType>> post_process_core(
    const std::vector<std::vector<size_t>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<QueryType> &queries,
    size_t ref_len, size_t stride, size_t k,
    std::function<std::pair<std::vector<std::string>, std::vector<ScoreType>>(
        const std::vector<std::string> &, const QueryType &, size_t)>
        rerank_func)
{
    size_t total_queries = queries.size();

    // Pre-allocate results for each query
    std::vector<std::vector<std::string>> all_final_seqs(total_queries);
    std::vector<std::vector<ScoreType>> all_scores(total_queries);

    // Thread-safe progress tracking
    std::atomic<size_t> completed_queries(0);

    // Hide cursor and create progress bar
    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"post-processing"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    // Parallelize across queries
    // #pragma omp parallel for num_threads(Config::PostProcess::NUM_THREADS) schedule(dynamic)
    for (size_t i = 0; i < total_queries; ++i)
    {
        //* Select only first 5 candidates by HNSW score as preliminary ranking
        std::vector<size_t> query_neighbors(neighbors[i].begin(),
                                            neighbors[i].begin() + std::min(size_t(5), neighbors[i].size()));

        // Get candidate sequences
        std::vector<std::string> query_cand_seqs = find_sequences(ref_genome, query_neighbors, ref_len, stride);

        // Rerank candidates using the provided function
        auto [top_seqs, top_scores] = rerank_func(query_cand_seqs, queries[i], k);

        // Store results for this query
        all_final_seqs[i] = std::move(top_seqs);
        all_scores[i] = std::move(top_scores);

        // Update progress bar (thread-safe)
        size_t current_completed = completed_queries.fetch_add(1) + 1;

        if (current_completed % 100 == 0)
        {
            size_t current_progress_percent = (current_completed * 100) / total_queries;
            progressBar.set_progress(current_progress_percent);
        }
    }

    // Flatten results
    std::vector<std::string> final_seqs;
    std::vector<ScoreType> scores;
    final_seqs.reserve(total_queries * k);
    scores.reserve(total_queries * k);

    for (size_t i = 0; i < total_queries; ++i)
    {
        final_seqs.insert(final_seqs.end(), all_final_seqs[i].begin(), all_final_seqs[i].end());
        scores.insert(scores.end(), all_scores[i].begin(), all_scores[i].end());
    }

    // Complete progress bar and show cursor
    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    return {final_seqs, scores};
}

/// @brief Post-process with Smith-Waterman reranking using dynamic sequence fetching
/// @param neighbors 2D vector of neighbor indices from HNSW search (size_t or long int)
/// @param distances 2D vector of distances from HNSW search
/// @param ref_genome Reference genome as a single string
/// @param query_seqs Vector of query sequences
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @param k Number of final candidates to return per query
/// @return Pair of vectors: 1st is sequences, 2nd is Smith-Waterman scores
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<int>> post_process_sw(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k);

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
/// @return Pair of vectors: 1st is sequences, 2nd is L2 distances
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_dynamic(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::string &ref_genome,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer);

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
/// @return Pair of vectors: 1st is sequences, 2nd is L2 distances
template <typename NeighborType>
std::pair<std::vector<std::string>, std::vector<float>> post_process_l2_static(
    const std::vector<std::vector<NeighborType>> &neighbors,
    const std::vector<std::vector<float>> &distances,
    const std::vector<std::string> &ref_seqs,
    const std::vector<std::string> &query_seqs,
    size_t ref_len, size_t stride, size_t k,
    const std::vector<std::vector<float>> &query_embeddings,
    Vectorizer &vectorizer);
