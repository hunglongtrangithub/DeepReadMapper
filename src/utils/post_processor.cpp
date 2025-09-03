#include "post_processor.hpp"

//! Only use for special HNSW index with encoded labels. Ignore if using default sequential labels.
PositionInfo position_decode(size_t label, const std::vector<size_t> &mapping)
{
    // First, check if label exists in mapping
    if (label >= mapping.size() || label < 0)
    {
        throw std::out_of_range("Label " + std::to_string(label) + " out of range for mapping of size " + std::to_string(mapping.size()));
    };

    // Then, get the encoded label
    size_t mapped_label = mapping[label];

    // Decode position and strand info
    return {
        mapped_label >> 1,      // Extract position by right-shifting 1 bit
        (mapped_label & 1) != 0 // Extract RC flag from lowest bit
    };
}

std::string find_sequence(const std::vector<std::string> &ref_seqs, int id, size_t ref_len)
{
    if (id < 0 || static_cast<size_t>(id) >= ref_seqs.size())
    {
        throw std::out_of_range("ID " + std::to_string(id) + " out of range for reference sequences of size " + std::to_string(ref_seqs.size()));
    }
    std::string cand_seq = ref_seqs[id];

    // Remove PREFIX and POSTFIX
    return cand_seq.substr(1, ref_len);
}

std::vector<std::string> find_sequences(const std::vector<std::string> &ref_seqs, const std::vector<size_t> &ids, size_t ref_len, size_t stride)
{
    std::vector<std::string> results;

    if (ids.empty())
        return results;

    // Step 1: Sort candidates and merge overlapping ranges
    std::vector<size_t> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    std::vector<std::pair<size_t, size_t>> ranges; // [start, end) pairs

    for (size_t base_id : sorted_ids)
    {
        size_t start = (base_id >= stride - 1) ? base_id - stride + 1 : 0;
        size_t end = std::min(base_id + stride, ref_seqs.size());

        // Merge with previous range if overlapping
        if (!ranges.empty() && start <= ranges.back().second)
        {
            ranges.back().second = std::max(ranges.back().second, end);
            continue;

            ranges.emplace_back(start, end);
        }

        // Step 2: Collect all unique positions from merged ranges
        std::vector<size_t> unique_positions;
        for (const auto &[start, end] : ranges)
        {
            for (size_t pos = start; pos < end; ++pos)
            {
                unique_positions.push_back(pos);
            }
        }

        // Step 3: Fetch sequences
        results.reserve(unique_positions.size());

        if (unique_positions.size() <= 100000)
        {
            for (size_t pos : unique_positions)
            {
                results.push_back(find_sequence(ref_seqs, pos, ref_len));
            }
        }
        else
        {
            results.resize(unique_positions.size());
#pragma omp parallel for num_threads(Config::Search::NUM_THREADS) schedule(static)
            for (size_t i = 0; i < unique_positions.size(); ++i)
            {
                results[i] = find_sequence(ref_seqs, unique_positions[i], ref_len);
            }
        }

        return results;
    }

    // TODO: Implement calc_sw_score and sm_rerank in post_processor.cpp
