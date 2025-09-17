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

    if (num_cands < k)
    {
        throw std::runtime_error("Not enough candidates (" + std::to_string(num_cands) + " < " + std::to_string(k) + ")");
    }

    top_seqs.reserve(k);
    top_scores.reserve(k);

    std::vector<size_t> indices(num_cands);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&scores](size_t i1, size_t i2)
                      { return scores[i1] > scores[i2]; });

    // Step 3: Collect top k sequences and scores
    for (size_t i = 0; i < k; ++i)
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

    if (num_cands < k)
    {
        throw std::runtime_error("Not enough candidates (" + std::to_string(num_cands) + " < " + std::to_string(k) + ")");
    }

    top_seqs.reserve(k);
    top_dists.reserve(k);

    std::vector<size_t> indices(num_cands);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&l2_dists](size_t i1, size_t i2)
                      { return l2_dists[i1] < l2_dists[i2]; });

    // Step 3: Collect top k sequences and distances
    for (size_t i = 0; i < k; ++i)
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

    std::cout << "[BATCH-RERANKER] Finished vectorizing " << all_cand_seqs.size() << " candidate sequences" << std::endl;

    // Step 2: Pre-allocate results vector
    std::vector<std::pair<std::vector<std::string>, std::vector<float>>> results(query_embeddings.size());

    // Progress tracking variables (thread-safe)
    std::atomic<size_t> completed_queries{0};
    const size_t total_queries = query_embeddings.size();

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
            results[q] = {{}, {}};

            // Update progress atomically
            size_t current_completed = completed_queries.fetch_add(1) + 1;
            if (current_completed % 1000 == 0 || current_completed == total_queries)
            {
#pragma omp critical
                {
                    size_t progress_percent = (current_completed * 100) / total_queries;
                    progressBar.set_progress(progress_percent);
                }
            }
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
        // There's an error if not enough candidates are available
        if (num_cands < k)
        {
            throw std::runtime_error("Not enough candidates (" + std::to_string(num_cands) + " < " + std::to_string(k) + ") for query " + std::to_string(q));
        }

        std::vector<size_t> indices(num_cands);
        std::iota(indices.begin(), indices.end(), 0);

        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&l2_dists](size_t i1, size_t i2)
                          { return l2_dists[i1] < l2_dists[i2]; });

        std::vector<std::string> top_seqs;
        std::vector<float> top_dists;
        top_seqs.reserve(k);
        top_dists.reserve(k);

        for (size_t i = 0; i < k; ++i)
        {
            top_seqs.push_back(all_cand_seqs[start_idx + indices[i]]);
            top_dists.push_back(l2_dists[indices[i]]);
        }

        results[q] = {top_seqs, top_dists};

        // Update progress atomically
        size_t current_completed = completed_queries.fetch_add(1) + 1;
        if (current_completed % 1000 == 0 || current_completed == total_queries)
        {
#pragma omp critical
            {
                size_t progress_percent = (current_completed * 100) / total_queries;
                progressBar.set_progress(progress_percent);
            }
        }
    }

    progressBar.set_progress(100);
    indicators::show_console_cursor(true);

    return results;
}