#include "reranker.hpp"

std::pair<std::vector<std::string>, std::vector<int>> reranker(const std::vector<std::string> &cand_seqs, const std::string &query_seq, size_t k)
{
    size_t num_cands = cand_seqs.size();
    if (num_cands == 0 || k == 0)
        return {{}, {}};

    // Step 1: Compute SW scores for all candidates
    std::vector<int> scores(num_cands, 0);

    for (size_t i = 0; i < num_cands; ++i)
    {
        scores[i] = calc_sw_score(cand_seqs[i], query_seq);
    }

    // Step 2: Sort candidates by score
    std::vector<std::string> top_seqs;
    std::vector<int> top_scores;

    size_t actual_k = std::min(k, num_cands);
    top_seqs.reserve(actual_k);
    top_scores.reserve(actual_k);

    std::vector<size_t> indices(num_cands);
    std::iota(indices.begin(), indices.end(), 0);

    // FIX: Only partial_sort up to actual_k, not k
    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
                      [&scores](size_t i1, size_t i2)
                      { return scores[i1] > scores[i2]; });

    // Step 3: Collect top actual_k sequences and scores
    for (size_t i = 0; i < actual_k; ++i)
    {
        top_seqs.push_back(cand_seqs[indices[i]]);
        top_scores.push_back(scores[indices[i]]);
    }

    return {top_seqs, top_scores};
}

std::pair<std::vector<std::string>, std::vector<float>> reranker(const std::vector<std::string> &cand_seqs, const std::vector<float> &query_embedding, size_t k, Vectorizer &vectorizer)
{
    size_t num_cands = cand_seqs.size();
    if (num_cands == 0 || k == 0)
        return {{}, {}};

    // Step 1: Compute embeddings for all candidates
    std::vector<std::vector<float>> cand_embeddings = vectorizer.vectorize(cand_seqs, false);
    std::vector<float> l2_dists;
    l2_dists.reserve(num_cands);
    for (const auto &cand_emb : cand_embeddings)
    {
        float dist = calc_l2_dist(cand_emb, query_embedding);
        l2_dists.push_back(dist);
    }

    // Step 2: Sort candidates by L2 distance
    std::vector<std::string> top_seqs;
    std::vector<float> top_dists;

    size_t actual_k = std::min(k, num_cands);
    top_seqs.reserve(actual_k);
    top_dists.reserve(actual_k);

    std::vector<size_t> indices(num_cands);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
                      [&l2_dists](size_t i1, size_t i2)
                      { return l2_dists[i1] < l2_dists[i2]; });

    // Step 3: Collect top actual_k sequences and distances
    for (size_t i = 0; i < actual_k; ++i)
    {
        top_seqs.push_back(cand_seqs[indices[i]]);
        top_dists.push_back(l2_dists[indices[i]]);
    }

    return {top_seqs, top_dists};
}

std::vector<std::pair<std::vector<std::string>, std::vector<float>>> batch_reranker(
    const std::vector<std::string> &all_cand_seqs,
    const std::vector<size_t> &query_start_indices,
    const std::vector<std::vector<float>> &query_embeddings,
    size_t k,
    Vectorizer &vectorizer)
{
    // Step 1: Single vectorization call for ALL candidates
    std::vector<std::vector<float>> all_cand_embeddings = vectorizer.vectorize(all_cand_seqs, false);

    // Step 2: Process each query's results
    //* Can be parallelized if needed
    std::vector<std::pair<std::vector<std::string>, std::vector<float>>> results;
    results.reserve(query_embeddings.size());

    indicators::show_console_cursor(false);
    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{80},
        indicators::option::PrefixText{"batch-reranking"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    #pragma omp parallel for num_threads(Config::PostProcess::NUM_THREADS) schedule(dynamic)
    for (size_t q = 0; q < query_embeddings.size(); ++q)
    {
        size_t start_idx = query_start_indices[q];
        size_t end_idx = query_start_indices[q + 1];
        size_t num_cands = end_idx - start_idx;

        if (num_cands == 0)
        {
            results.push_back({{}, {}});
            continue;
        }

        // Calculate L2 distances for this query
        std::vector<float> l2_dists;
        l2_dists.reserve(num_cands);
        for (size_t i = start_idx; i < end_idx; ++i)
        {
            float dist = calc_l2_dist(all_cand_embeddings[i], query_embeddings[q]);
            l2_dists.push_back(dist);
        }

        // Sort and get top k
        size_t actual_k = std::min(k, num_cands);
        std::vector<size_t> indices(num_cands);
        std::iota(indices.begin(), indices.end(), 0);

        std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
                          [&l2_dists](size_t i1, size_t i2)
                          { return l2_dists[i1] < l2_dists[i2]; });

        std::vector<std::string> top_seqs;
        std::vector<float> top_dists;
        top_seqs.reserve(actual_k);
        top_dists.reserve(actual_k);

        for (size_t i = 0; i < actual_k; ++i)
        {
            top_seqs.push_back(all_cand_seqs[start_idx + indices[i]]);
            top_dists.push_back(l2_dists[indices[i]]);
        }

        results.push_back({top_seqs, top_dists});

        // Update progress bar
        if (q % 100 == 0 || q == query_embeddings.size() - 1)
        {
            size_t progress_percent = (q * 100) / query_embeddings.size();
            progressBar.set_progress(progress_percent);
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    return results;
}