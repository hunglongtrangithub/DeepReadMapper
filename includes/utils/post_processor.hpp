#include "utils.hpp"
#include "config.hpp"
#include <omp.h>

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

///@brief Compute the Smith-Waterman alignment score between two sequences
///@param seq1 First sequence
///@param seq2 Second sequence
///@return Smith-Waterman alignment score
int calc_sw_score(const std::string &seq1, const std::string &seq2);

/// @brief A reranker based on SM-score that takes in K * 2 * stride candidates and returns top K candidates as (labels, dists) pairs
/// @param cand_seqs Vector of candidate sequences (removed PREFIX/POSTFIX)
/// @param query_seq Query sequence to align against (removed PREFIX/POSTFIX)
/// @param K Number of final candidates to return
/// @param stride Stride used during FASTA preprocessing

std::pair<std::vector<size_t>, std::vector<float>> sm_rerank(const std::vector<std::string> &cand_seqs, const std::string &query_seq, size_t K, size_t stride);
