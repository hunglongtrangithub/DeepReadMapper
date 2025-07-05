#include "config.hpp"
#include "hnswlib.h"

hnswlib::HierarchicalNSW<float> *index()
{
    // Build parameters
    int dim = Config::Build::DIM;
    int MAX_ELE = Config::Build::MAX_ELEMENTS;
    int M = Config::Build::GPH_DEG;
    int EFC = Config::Build::EFC;

    // Initialize index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, MAX_ELE, M, EFC);

    // Generate and add data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float *data = new float[dim * MAX_ELE];
    for (int i = 0; i < dim * MAX_ELE; i++)
    {
        data[i] = distrib_real(rng);
    }

    for (int i = 0; i < MAX_ELE; i++)
    {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    return alg_hnsw;
}

int search(hnswlib::HierarchicalNSW<float> *alg_hnsw)
{
    int ef = Config::Search::EF;
    int k = Config::Search::K;
    int dim = Config::Build::DIM;
    int MAX_ELE = Config::Build::MAX_ELEMENTS;

    alg_hnsw->setEf(ef);

    // Generate test data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float *data = new float[dim * MAX_ELE];
    for (int i = 0; i < dim * MAX_ELE; i++)
    {
        data[i] = distrib_real(rng);
    }

    // Query with correct parameters
    float correct = 0;
    for (int i = 0; i < MAX_ELE; i++)
    {
        // ✅ USE CONFIG VALUES INSTEAD OF HARDCODED
        auto result = alg_hnsw->searchKnn(data + i * dim, k);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }

    float recall = correct / MAX_ELE;
    std::cout << "Recall: " << recall << "\n";

    delete[] data;
    return 0;
}