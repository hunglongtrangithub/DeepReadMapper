#include "utils.hpp"
#include "config.hpp"
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

struct PositionInfo
{
    size_t position; // Actual position in the original sequence (FASTA)
    bool RC;         // 0 for forward strand, 1 for reverse complement
};

/// @brief Decode position and strand information from a label
/// @param label HNSWPQ sequential label
/// @param mapping Custom vector that map sequential label to original position
/// @return PositionInfo struct containing position and RC flag
PositionInfo position_decode(size_t label, const std::vector<size_t> &mapping);

/// @brief Reformat raw HNSW output (vector of pairs) into two separate vectors
/// @param neighbors Vector of vectors containing neighbor indices as size_t
/// @param distances Vector of vectors containing L2 distances from query to neighbors
/// @return Pair of vectors: 1st is labels, 2nd is distances
std::pair<std::vector<size_t>, std::vector<float>> reformat_output(const std::vector<std::vector<size_t>> &neighbors, const std::vector<std::vector<float>> &distances, size_t k);

/// @brief Overload reformat function for long int type
std::pair<std::vector<size_t>, std::vector<float>> reformat_output(const std::vector<std::vector<long int>> &neighbors, const std::vector<std::vector<float>> &distances, size_t k);

/// @brief Find full-length sequence from a set of reference sequences
/// @param ref_seqs Vector of reference sequences (with PREFIX/POSTFIX)
/// @param id Id of the sequence to find
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @return Index of the sequence in ref_seqs, or -1 if not found
std::string find_sequence(const std::vector<std::string> &ref_seqs, size_t id, size_t ref_len);

/// @brief Wrapper to find multiple sequences asynchronously
/// @param ref_seqs Vector of reference sequences (with PREFIX/POSTFIX)
/// @param ids Vector of ids to find
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @return Vector of sequences found, in the same order as ids
std::vector<std::string> find_sequences(const std::vector<std::string> &ref_seqs, const std::vector<size_t> &ids, size_t ref_len, size_t stride = 1);

///@brief AVX2 optimized Smith-Waterman alignment score computation
///@param seq1 First sequence
///@param seq2 Second sequence
///@return Smith-Waterman alignment score
int calc_sw_score_avx2(const std::string &seq1, const std::string &seq2);

///@brief Compute the Smith-Waterman alignment score between two sequences
///@param seq1 First sequence
///@param seq2 Second sequence
///@return Smith-Waterman alignment score
int calc_sw_score(const std::string &seq1, const std::string &seq2);

/// @brief A reranker that takes in EF * 2 * stride candidates and returns top k candidates as (seq, dists) pairs. Multiple modes supported (SW-score, L2-dist, etc.)
/// @param cand_seqs Vector of candidate sequences (removed PREFIX/POSTFIX)
/// @param query_seq Query sequence to align against (removed PREFIX/POSTFIX)
/// @param k Number of final candidates to return
/// @param stride Stride used during FASTA preprocessing
/// @return Pair of vectors: 1st is labels, 2nd is edit distances (SW-score)

/// @brief Calculate L2 distance between two vectors (or euclidean distance)
/// @param vec1 First vector
/// @param vec2 Second vector
/// @return L2 distance
float calc_l2_dist(const std::vector<float> &vec1, const std::vector<float> &vec2);

std::pair<std::vector<std::string>, std::vector<int>> sw_reranker(const std::vector<std::string> &cand_seqs, const std::string &query_seq, size_t k, size_t stride);

/// @brief A wrapper to handle the post-processing step after HNSW search
/// @param neighbors 2D vector of neighbor indices from HNSW search
/// @param distances 2D vector of distances from HNSW search
/// @param ref_seqs Vector of reference sequences (with PREFIX/POSTFIX)
/// @param query_seqs Vector of query sequences (with PREFIX/POSTFIX)
/// @param ref_len Length of each reference sequence, doesn't include PREFIX/POSTFIX
/// @param stride In case of FASTA preprocessing with stride, used to compute actual positions
/// @param k Number of final candidates to return per query
/// @return Pair of vectors: 1st is labels, 2nd is edit distances
std::pair<std::vector<std::string>, std::vector<int>> post_process(const std::vector<std::vector<size_t>> &neighbors, const std::vector<std::vector<float>> &distances, const std::vector<std::string> &ref_seqs, const std::vector<std::string> &query_seqs, size_t ref_len, size_t stride, size_t k);

/// @brief An overload for post_process to accept long int type for neighbors
std::pair<std::vector<std::string>, std::vector<int>> post_process(const std::vector<std::vector<long int>> &neighbors, const std::vector<std::vector<float>> &distances, const std::vector<std::string> &ref_seqs, const std::vector<std::string> &query_seqs, size_t ref_len, size_t stride, size_t k);