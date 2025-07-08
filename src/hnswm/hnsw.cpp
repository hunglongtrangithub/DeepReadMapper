#include "hnsw.h"
#include "progressbar.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <numeric>
#include <assert.h>
#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/optional.hpp>
#include <immintrin.h>
#include <omp.h>

#define ENTRYPOINT_NOT_SET 0xFFFFFFFF

bool _Profiling_enabled = false;
uint64_t _Count_dist_calc = 0;

void enableProfiling()
{
    _Profiling_enabled = true;
}

void disableProfiling()
{
    _Profiling_enabled = false;
}

uint64_t getCountDistCalc()
{
    return _Count_dist_calc;
}

void resetProfilingCounter()
{
    _Count_dist_calc = 0;
}

// float squaredEuclideanDistance(const std::vector<float> &v1, const std::vector<float> &v2)
// {
//     float distance = 0.0f;
//     for (size_t i = 0; i < v1.size(); ++i)
//     {
//         float diff = v1[i] - v2[i];
//         distance += diff * diff;
//     }
//     return distance;
// }

// float squaredEuclideanDistance(const std::vector<float> &v1, const std::vector<float> &v2)
// {
//     __m128 vd2 = _mm_set1_ps(0.0f);
//     float d2 = 0.0f;
//     unsigned k;

//     // process 4 elements per iteration
//     for (k = 0; k < v1.size() - 3; k += 4)
//     {
//         __m128 va = _mm_load_ps(&v1[k]);
//         __m128 vb = _mm_load_ps(&v2[k]);
//         __m128 vd = _mm_sub_ps(va, vb);
//         vd = _mm_mul_ps(vd, vd);
//         vd2 = _mm_add_ps(vd2, vd);
//     }

//     // horizontal sum of 4 partial dot products
//     vd2 = _mm_hadd_ps(vd2, vd2);
//     vd2 = _mm_hadd_ps(vd2, vd2);
//     _mm_store_ss(&d2, vd2);
//     return d2;
// }

float squaredEuclideanDistance(const std::vector<float> &v1, const std::vector<float> &v2)
{
    if (_Profiling_enabled) _Count_dist_calc++;
    assert (v1.size() == VECTOR_DIM);
    assert (v2.size() == VECTOR_DIM);

    __m256 sum = _mm256_setzero_ps();

    // Process 8 floats at a time using AVX2
    for (unsigned i = 0; i < VECTOR_DIM; i += 8)
    {
        __m256 vec1 = _mm256_loadu_ps(&v1[i]);
        __m256 vec2 = _mm256_loadu_ps(&v2[i]);
        __m256 diff = _mm256_sub_ps(vec1, vec2);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Sum up all elements
    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum);
    return temp[3] + temp[4];
}

NSW::NSW(uint64_t maxNumNodes)
{
    this->maxNumNodes_ = maxNumNodes;
    this->numNodes_ = 0;
    this->neighbors_ = std::vector<std::optional<std::vector<uint32_t>>>(maxNumNodes);
    this->defaultEntryID_ = ENTRYPOINT_NOT_SET;
}

uint32_t NSW::getEntrypointID() const
{
    if (this->numNodes_ == 0)
    {
        throw std::runtime_error("this NSW graph is empty!");
    }
    return this->defaultEntryID_;
}

uint32_t NSW::getNeighborhoodSize(nodeID_t nodeID)
{
    auto &neighborhood = this->neighbors_[nodeID];
    if (!neighborhood.has_value())
        throw std::runtime_error("asked for neighborhood size but node does not exist!");
    return this->neighbors_[nodeID]->size();
}

const std::vector<nodeID_t> &NSW::getNeighborsId(nodeID_t nodeID) const
{
    return *(this->neighbors_[nodeID]);
}

bool NSW::isEmpty() const
{
    return this->numNodes_ == 0;
}

uint32_t NSW::getNumNodes() const
{
    return this->numNodes_;
}

bool NSW::containsNode(nodeID_t nodeID) const
{
    return this->neighbors_[nodeID] != std::nullopt;
}

void NSW::addNode(uint32_t nodeID, const std::vector<uint32_t> &neighborsNodeIds)
{
    // sanity check that this nodeID has not been added before
    if (this->containsNode(nodeID))
    {
        throw std::runtime_error("nodeID " + std::to_string(nodeID) + " already exist in this NSW graph, can't add again");
    }
    // sanity check capacity
    if (this->numNodes_ >= this->maxNumNodes_)
    {
        throw std::runtime_error("Capacity is maxed out, please increase capacity when initialize object");
    }
    if (nodeID >= this->maxNumNodes_)
    {
        throw std::runtime_error("nodeID out of bound. Please increase HNSW's max num nodes.");
    }
    // increment nodecount
    this->numNodes_++;
    // add connections from this node to the provided neighbors
    this->neighbors_[nodeID] = neighborsNodeIds;
    // add connections from the provided neighbors to this node
#pragma GCC unroll 16
    for (auto &neighborID : neighborsNodeIds)
    {
        if (!this->containsNode(neighborID))
        {
            throw std::runtime_error("Can't connect a new node to a non-existing neighbor");
        }
        this->neighbors_[neighborID]->push_back(nodeID);
    }
    // if entrypointID not set (first node being added), set it to this first node
    if (this->defaultEntryID_ == ENTRYPOINT_NOT_SET)
    {
        this->defaultEntryID_ = nodeID;
    }
}

void NSW::removeConnection(nodeID_t nodeID1, nodeID_t nodeID2)
{
    if ((!this->containsNode(nodeID1)) || (!this->containsNode(nodeID2)))
    {
        throw std::runtime_error("trying to remove connections of a node that does not exist");
    }
    auto postRemoveIter = std::remove(
        this->neighbors_[nodeID1]->begin(),
        this->neighbors_[nodeID1]->end(), nodeID2);
    this->neighbors_[nodeID1]->erase(postRemoveIter, this->neighbors_[nodeID1]->end());
    postRemoveIter = std::remove(
        this->neighbors_[nodeID2]->begin(),
        this->neighbors_[nodeID2]->end(),
        nodeID1);
    this->neighbors_[nodeID2]->erase(postRemoveIter, this->neighbors_[nodeID2]->end());
}

void NSW::summarize() const
{
    std::cout << "Graph has " << this->numNodes_ << " nodes. ";

    if (!(this->isEmpty()))
    {
        // Calculate min, max, and average number of neighbors per node
        size_t totalNeighbors = 0;
        size_t minNeighbors = 0xfffffff;
        size_t maxNeighbors = 0;
        for (const auto &neighborList : this->neighbors_)
        {
            if (!neighborList.has_value())
                continue;
            size_t numNeighbors = neighborList->size();
            totalNeighbors += numNeighbors;
            if (numNeighbors < minNeighbors)
            {
                minNeighbors = numNeighbors;
            }
            if (numNeighbors > maxNeighbors)
            {
                maxNeighbors = numNeighbors;
            }
        }
        double averageNeighbors = static_cast<double>(totalNeighbors) / numNodes_;
        std::cout << "Neighbors per node: min=" << minNeighbors << " ";
        std::cout << "avg=" << averageNeighbors << " ";
        std::cout << "max=" << maxNeighbors << "\n";
    }
}

void NSW::printNodes() const
{
    if (this->numNodes_ < 20)
    {
        // print all nodes
        for (nodeID_t nodeID = 0; nodeID < maxNumNodes_; nodeID++)
        {
            if (this->containsNode(nodeID))
                std::cout << nodeID << " ";
        }
    }
    else
    {
        // print first 10
        for (nodeID_t nodeID = 0, count = 0; nodeID < this->maxNumNodes_; nodeID++)
        {
            if (this->containsNode(nodeID))
            {
                std::cout << nodeID << " ";
                count++;
                if (count == 10)
                    break;
            }
        }
        std::cout << " ... ";
        // print last 10
        for (nodeID_t nodeID = this->maxNumNodes_ - 1, count = 0; nodeID >= 0; nodeID--)
        {
            if (this->containsNode(nodeID))
            {
                std::cout << nodeID << " ";
                if (count == 10)
                    break;
            }
        }
    }
    std::cout << std::endl;
}

// insert into a sorted increasing array in logN time to keep it sorted
template <typename T>
static inline void insertSorted(std::vector<T> &arr, T newObj)
{
    auto it = std::lower_bound(arr.begin(), arr.end(), newObj);
    arr.insert(it, newObj);
}

HNSW::HNSW(uint32_t dim, uint32_t efc, uint32_t M, uint64_t maxNumNodes)
{
    this->maxNumNodes_ = maxNumNodes;
    this->mL_ = 1 / log(M);
    this->efc_ = (efc > M) ? efc : M;
    this->layers = std::vector<NSW>(NSW_LAYERS_PREALLOC, NSW(maxNumNodes));
    this->nLayers_ = 0;
    this->M_ = M;
    this->Mmax_ = M;
    this->Mmax0_ = M * 2;
    this->prepareMemory();
}

void HNSW::prepareMemory()
{
    this->searchCandidates_.reserve(80000);
    this->searchVisited_.reserve(this->maxNumNodes_);
    this->searchDists_.resize(this->Mmax0_);
}

const std::vector<float> &HNSW::getVector_(nodeID_t nodeID) const
{
    return this->vectors_[nodeID];
}

NSW &HNSW::getTopLayer_()
{
    for (int i = this->layers.size() - 1; i > -1; i--)
    {
        if (!(this->layers[i].isEmpty()))
        {
            return this->layers[i];
        }
    }
    throw std::runtime_error("All layers in this HNSW are empty");
}

void NSW::findAndSetDefaultEntry()
{
    uint64_t maxDegree = 0;
    for (nodeID_t i = 0; i < this->maxNumNodes_; i++)
    {
        if (!this->containsNode(i))
            continue;
        auto degree = this->neighbors_[i]->size();
        if (degree > maxDegree)
        {
            maxDegree = degree;
            this->defaultEntryID_ = i;
        }
    }
}

std::vector<uint32_t> HNSW::calculateNumNodesInLayers_(uint32_t total) const
{
    double mL = this->mL_;
    auto cdf = [mL](double x)
    { return 1 - std::exp(-x / mL); };
    std::vector<uint32_t> expected(10000);
    for (unsigned k = 0; k < 10000; ++k)
    {
        expected[k] = round((cdf(static_cast<double>(k + 1)) - cdf(static_cast<double>(k))) * total);
    }
    // remove the 0 elements
    expected.erase(std::remove_if(expected.begin(), expected.end(), [](uint32_t e)
                                  { return e == 0; }),
                   expected.end());
    // adjust the last element so that the sum is exactly = `total`
    expected.back() = total - std::accumulate(expected.begin(), expected.end() - 1, 0);
    assert(std::accumulate(expected.begin(), expected.end(), uint32_t(0)) == total);
    return expected;
}

float HNSW::calcDistBtw2Ids_(nodeID_t nodeID1, nodeID_t nodeID2) const
{
    auto node1Vec = this->getVector_(nodeID1);
    auto node2Vec = this->getVector_(nodeID2);
    return DIST_FUNC(node1Vec, node2Vec);
}

float HNSW::calcDistToId_(const std::vector<float> &query, nodeID_t nodeID) const
{
    auto nodeVec = this->getVector_(nodeID);
    return DIST_FUNC(query, nodeVec);
}

nodeID_t HNSW::findCentroid_(nodeID_t start, nodeID_t end) const
{
    // calculate the centroid vector
    std::vector<float> centroid(VECTOR_DIM, 0);
    for (auto i = start; i < end; i++)
    {
        for (unsigned j = 0; j < VECTOR_DIM; j++)
        {
            centroid[j] += this->getVector_(i)[j];
        }
    }
    auto numVectors = end - start;
    for (unsigned j = 0; j < VECTOR_DIM; j++)
    {
        centroid[j] /= numVectors;
    }
    // calculate distances to centroid
    std::vector<float> distances(numVectors, 0);
    for (auto i = start; i < end; i++)
    {
        distances[i - start] = DIST_FUNC(this->getVector_(i), centroid);
    }
    // the index of the min element
    auto minDistIdx = std::min_element(distances.begin(), distances.end()) - distances.begin();

    return start + minDistIdx;
}

void HNSW::selectNeighborsSimple_(
    const std::vector<float> &query, std::vector<search_result_t> &candidates, uint32_t M) const
{
    candidates.resize(M);
    return;
}

void HNSW::selectNeighborsHeuristic_(
    const std::vector<float> &query, std::vector<search_result_t> &candidates, uint32_t M, uint32_t layerID)
{
    // min heap (distance, nodeID)
    DaryHeap<4, search_result_t, std::greater<search_result_t>> workingQueue(8000);
    // add candidates and surrounding neighbors to the working queue
    auto& visited = this->searchVisited_;
    visited.clear();
    for (const auto &candidate : candidates)
    {
        // add candidates
        auto distNodeToQuery = candidate.first;
        auto nodeID = candidate.second;
        workingQueue.push(std::make_pair(distNodeToQuery, nodeID));
        visited.insert(nodeID);
        // add surrounding neighbors
        auto neighborIDs = this->layers[layerID].getNeighborsId(nodeID);
        for (auto nbID : neighborIDs)
        {
            if (!visited.contains(nbID))
            {
                float nbDistToQuery = this->calcDistToId_(query, nbID);
                workingQueue.push(std::make_pair(nbDistToQuery, nbID));
                visited.insert(nbID);
            }
        }
    }
    // now clear the candidate list and repopulate it
    candidates.resize(0);
    // min heap (distance, nodeID)
    DaryHeap<4, search_result_t, std::greater<search_result_t>> discarded(8000);
    while (workingQueue.size() > 0 && candidates.size() < M)
    {
        auto distCandidateToQuery = workingQueue.top().first;
        auto candidateID = workingQueue.top().second;
        workingQueue.pop();
        bool good = true;
        for (const auto &[dist, addedNodeID] : candidates)
        {
            float distCandidateToAddedNode = this->calcDistBtw2Ids_(candidateID, addedNodeID);
            if (distCandidateToAddedNode < distCandidateToQuery)
            {
                good = false;
                break;
            }
        }
        // we can add this candidateID to the result or discard
        if (good)
        {
            candidates.push_back(std::make_pair(distCandidateToQuery, candidateID));
        }
        else
        {
            discarded.push(std::make_pair(distCandidateToQuery, candidateID));
        }
    }
    // Add some discarded candidates if the return list is not large enough
    while (candidates.size() < M && discarded.size() > 0)
    {
        auto distCandidateToQuery = discarded.top().first;
        auto candidateID = discarded.top().second;
        discarded.pop();
        candidates.push_back(std::make_pair(distCandidateToQuery, candidateID));
    }
    return;
}

void HNSW::insertLayer_(uint32_t layerID, const std::vector<float> &vector, nodeID_t nodeID)
{
    // first we find the entrypoints of the top layers above layerID
    nodeID_t entrypointID = ENTRYPOINT_NOT_SET;    // use the default entrypoint on NSW layers
    std::vector<search_result_t> nearestNeighbors; // Sorted array (distance, nodeID) increasing
    for (unsigned l = this->layers.size() - 1; l > layerID; --l)
    {
        if (layers[l].isEmpty())
            continue;
        // search the current layer, starting from the best node found in the previous layer
        nearestNeighbors = this->searchLayer(l, entrypointID, vector, 1);
        entrypointID = nearestNeighbors[0].second;
    }
    // now we add this vector to the layers from layerID to below
    for (int l = layerID; l >= 0; --l)
    {
        NSW &layerGraph = layers[l];
        // find connecting neighbors for this node in this layer
        nearestNeighbors = this->searchLayer(l, entrypointID, vector, this->M_ * 2);
        this->selectNeighborsHeuristic_(vector, nearestNeighbors, this->M_, l);
        std::vector<nodeID_t> neighborsID;
        for (const auto &[_dist, nID] : nearestNeighbors)
        {
            neighborsID.push_back(nID);
        }
        layerGraph.addNode(nodeID, neighborsID);
        // the neighbors' degrees just increased by 1
        // check neighbors' degrees and shrink their connections if degree > Mmax
        auto Mmax = (l == 0) ? this->Mmax0_ : this->Mmax_;
        for (auto neighborID : neighborsID)
        {
            if (layerGraph.getNeighborhoodSize(neighborID) > Mmax)
            {
                auto neighbors2ID = layerGraph.getNeighborsId(neighborID);
                // max distance from a 2nd neighbor to this neighborID
                float maxDist = -1;
                // ID of the 2nd neighbor with max distance
                nodeID_t maxNeighbor2ID = neighbors2ID.front();
                for (auto n2ID : neighbors2ID)
                {
                    float nToN2Dist = this->calcDistBtw2Ids_(neighborID, n2ID);
                    if (nToN2Dist > maxDist)
                    {
                        maxDist = nToN2Dist;
                        maxNeighbor2ID = n2ID;
                    }
                }
                layerGraph.removeConnection(neighborID, maxNeighbor2ID);
            }
        }
    }
    return;
}

typedef struct kernel_data_t
{
    const std::vector<float> *query;
    const std::vector<std::vector<float>> *db;
    const nodeID_t *neighborIDs;
    unsigned neighborIDsLen;
    VisitedCheck *visited;
    float* output;
} kernel_data_t;

void *kernel(void *data)
{
    kernel_data_t input = *((kernel_data_t*)data);
    for (unsigned i=0; i < input.neighborIDsLen; i++)
    {
        // if (i % 16 == 0)
        //     _mm_prefetch(reinterpret_cast<const char*>(&this->searchDists_[i+16]), _MM_HINT_T0);
        nodeID_t neighborID = input.neighborIDs[i];
        if (input.visited->contains(neighborID))
        {
            input.output[i] = -1;
            continue;
        };
        input.visited->insert(neighborID);
        const std::vector<float>& vec = (*input.db)[neighborID];
        input.output[i] = DIST_FUNC(*input.query, vec);
    }
    return nullptr;
}

#define PRECOMP_NTHREADS 1
void precomputeDistances(
    const std::vector<float> &query, const std::vector<std::vector<float>> &db, const std::vector<nodeID_t> &neighborIDs, VisitedCheck &visited, std::vector<float> &outDist)
{
    // // pthread_t threads[PRECOMP_NTHREADS];
    // unsigned nbPerThread = neighborIDs.size() / PRECOMP_NTHREADS;
    // unsigned beg = 0;
    for (unsigned i=0; i < neighborIDs.size(); i++)
    {
        if (i % 16 == 0)
            _mm_prefetch(reinterpret_cast<const char*>(&outDist[i+16]), _MM_HINT_T0);
        nodeID_t neighborID = neighborIDs[i];
        if (visited.contains(neighborID))
        {
            outDist[i] = -1;
            continue;
        };
        visited.insert(neighborID);
        outDist[i] = DIST_FUNC(query, db[neighborID]);
    }

    // for (unsigned i = 0; i < PRECOMP_NTHREADS; ++i)
    //     pthread_join(threads[i], nullptr);
}

std::vector<search_result_t> HNSW::searchLayer(
    uint32_t layerID, nodeID_t entrypointID, const std::vector<float> &query, uint32_t ef)
{
    const NSW &layerGraph = this->layers[layerID];
    if (layerGraph.isEmpty())
        return std::vector<search_result_t>(0);
    if (entrypointID == ENTRYPOINT_NOT_SET)
        entrypointID = layerGraph.getEntrypointID();
    // visited nodes on this NSW graph
    auto& visited = this->searchVisited_;
    visited.clear();
    visited.insert(entrypointID);
    // distance from query to entrypoint
    float entrypointDist = this->calcDistToId_(query, entrypointID);
    // Candidate list. min heap (distance, nodeID)
    auto& candidates = this->searchCandidates_;
    candidates.clear();
    candidates.push(std::make_pair(entrypointDist, entrypointID));
    // sorted array of (distance, nodeID) increasing
    std::vector<search_result_t> nearestNeighbors{{entrypointDist, entrypointID}};
    nearestNeighbors.reserve(ef + 1);
    while (!candidates.empty())
    {
        // consider the closest candidate to the query
        auto [candidateDist, candidateID] = candidates.top();
        candidates.pop();
        // stop search if candidate is further than the current worst on nearestNeighbors
        if (candidateDist > nearestNeighbors.back().first)
            break;
        auto& neighborIDs = layerGraph.getNeighborsId(candidateID);
        // pre-compute distances to all neighbors
        precomputeDistances(query, this->vectors_, neighborIDs, visited, this->searchDists_);
        // check all neighbors of this candidate
        for (unsigned i=0; i<neighborIDs.size(); i++)
        {
            float neighborDist;
            if (i % 16 == 12)
                _mm_prefetch(reinterpret_cast<const char*>(&this->searchDists_[i+16]), _MM_HINT_T0);
            // distance from query to this neighbor
            neighborDist = this->searchDists_[i];
            if (neighborDist == -1)
                continue;
            // if this neighbor distance is better than the worst on nearestNeighbors,
            // put it on both the candidates and the nearestNeighbors lists
            // if the nearestNeighbors list is full, remove the furthest (last) node
            if ((neighborDist < nearestNeighbors.back().first) || nearestNeighbors.size() < ef)
            {
                nodeID_t neighborID = neighborIDs[i];
                // update candidate list
                candidates.push(std::make_pair(neighborDist, neighborID));
                // update nearest neighbors list
                insertSorted(nearestNeighbors, std::make_pair(neighborDist, neighborID));
                if (nearestNeighbors.size() > ef)
                    nearestNeighbors.pop_back();
            }
        }
    }
    return nearestNeighbors;
}

std::vector<search_result_t> HNSW::search(const std::vector<float> &query, uint32_t ef)
{
    if (this->layers[0].isEmpty())
    {
        return {};
    }
    if (query.size() != this->vectors_[0].size())
        throw std::invalid_argument("query does not have correct dimensions");
    nodeID_t entrypointID = ENTRYPOINT_NOT_SET; // Use default entry point
    for (uint32_t layerID = NSW_LAYERS_PREALLOC - 1; layerID > 0; layerID--)
    {
        if (this->layers[layerID].isEmpty())
        {
            continue;
        }
        // Search the current layer, starting from the best node found in the previous layer
        auto nearestNeighbors = this->searchLayer(layerID, entrypointID, query, 1);
        entrypointID = nearestNeighbors[0].second;
    }
    return this->searchLayer(0, entrypointID, query, ef);
}

std::vector<std::vector<search_result_t>> HNSW::searchParallel(
    const std::vector<std::vector<float>> &queries, uint32_t ef)
{
    if (this->layers[0].isEmpty())
    {
        return {};
    }
    if (queries[0].size() != this->vectors_[0].size())
        throw std::invalid_argument("query 0 does not have correct dimensions");
    std::vector<std::vector<search_result_t>> output(queries.size());
#pragma omp parallel for shared(queries, ef, output)
    for (uint32_t i = 0; i < queries.size(); i++)
    {
        const auto &query = queries[i];
        nodeID_t entrypointID = ENTRYPOINT_NOT_SET; // Use default entry point
        for (uint32_t layerID = NSW_LAYERS_PREALLOC - 1; layerID > 0; layerID--)
        {
            if (this->layers[layerID].isEmpty())
            {
                continue;
            }
            // Search the current layer, starting from the best node found in the previous layer
            auto nearestNeighbors = this->searchLayer(layerID, entrypointID, query, 1);
            entrypointID = nearestNeighbors[0].second;
        }
        output[i] = this->searchLayer(0, entrypointID, query, ef);
    }
    return output;
}

uint32_t HNSW::getNumLayers()
{
    if (this->nLayers_ > 0) return this->nLayers_;
    uint32_t n = 0;
    for (uint32_t i = 0; i < this->layers.size(); i++)
    {
        if (this->layers[i].isEmpty())
            break;
        n = i;
    }
    this->nLayers_ = n;
    return n;
}

void HNSW::buildIndex(const std::vector<std::vector<float>> &vectors)
{
    if (this->vectors_.size() + vectors.size() > this->maxNumNodes_)
        throw std::runtime_error("Capacity maxed out. Please increase max num nodes.");
    // Add vectors to db
    uint32_t nodeIDOffset = vectors_.size();
    this->vectors_.insert(
        this->vectors_.end(),
        vectors.begin(), vectors.end());
    // Calculate number of vectors that go into each layer
    std::vector<uint32_t> layerSizes = this->calculateNumNodesInLayers_(vectors.size());
    // Add vectors to each layer from top to bottom
    // This array contains partition endpoints (a, b), (b, c) where a, b, c are index on vectors. Endpoints are excluded from partitions
    std::vector<int> partitions = {-1, static_cast<int>(vectors.size())};
    for (int layerID = layerSizes.size() - 1; layerID >= 0; --layerID)
    {
        auto nPoints = layerSizes[layerID];
        // setup progress bar
        indicators::show_console_cursor(false);
        indicators::ProgressBar progressBar{
            indicators::option::BarWidth{80},
            indicators::option::PrefixText{
                "layer " + std::to_string(layerID) + " adding " + std::to_string(nPoints) + " nodes"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};
        uint32_t pointsAdded = 0;
        // Number of current partitions
        uint32_t nParts = partitions.size() - 1;
        // Number of points to add per current partition
        uint32_t pointsPerPart = static_cast<uint32_t>(round(static_cast<float>(nPoints) / nParts));
        // For each current partition,
        // split the partition equally into pointsPerPart sub-partitions
        // For each sub-partition, get the centroid and add to layer
        std::vector<int> newPartitions;
        for (uint32_t i = 0; i < partitions.size() - 1; ++i)
        {
            int partBegIdx = partitions[i];
            int partEndIdx = partitions[i + 1];
            uint32_t partSize = partEndIdx - partBegIdx - 1;
            std::vector<uint32_t> selectedPtsIdx;
            if (layerID == 0 || partSize < pointsPerPart)
            {
                // select all points in this partition
                selectedPtsIdx.resize(partEndIdx - partBegIdx - 1);
                auto it = selectedPtsIdx.begin();
                uint32_t value = partBegIdx + 1;
#pragma GCC unroll 4
                for (; it != selectedPtsIdx.end(); ++it, ++value)
                {
                    *it = value;
                }
            }
            else
            {
                // find and select the centroid of each sub-partition
                // size of each sub-partition
                uint32_t subpartSize = partSize / pointsPerPart;
                // iterate to create sub-partitions endpoints. There are pointsPerPart-1 endpoints
                int previousEndpoint = partBegIdx + 1;
                int currentEndpoint;
                for (uint32_t j = 0; j < pointsPerPart; ++j)
                {
                    // the endpoint for this subpartition
                    currentEndpoint = (j == (pointsPerPart - 1)) ? partEndIdx : previousEndpoint + subpartSize;
                    assert(currentEndpoint <= partEndIdx);
                    // find the index of the centroid of the subpartition (previousEndpoint, currentEndPoint)
                    selectedPtsIdx.push_back(this->findCentroid_(previousEndpoint, currentEndpoint));
                    previousEndpoint = currentEndpoint;
                }
            }
            // Now add each selected points to layerID
            for (auto &idx : selectedPtsIdx)
            {
                // Add vector to database and get a new node ID
                nodeID_t nodeID = nodeIDOffset + idx;
                this->insertLayer_(layerID, vectors[idx], nodeID);
                // update progress bar
                pointsAdded++;
                float newProgressCompleted = static_cast<float>(pointsAdded) / nPoints * 100;
                progressBar.set_progress((newProgressCompleted < 100) ? newProgressCompleted : 99);
            }
            // Update partitions
            newPartitions.push_back(partBegIdx);
            newPartitions.insert(newPartitions.end(), selectedPtsIdx.begin(), selectedPtsIdx.end());
        }
        // Done with this layerID, update partitions
        newPartitions.push_back(static_cast<int>(vectors.size()));
        partitions = newPartitions;
        // Close Current progress bar
        progressBar.mark_as_completed();
        indicators::show_console_cursor(true);
    }
    // // index building finished. Now find the entrypoint for the top layer
    // auto& topLayer = this->getTopLayer_();
    // topLayer.findAndSetDefaultEntry();
}

void HNSW::summarize() const
{
    std::cout << "================HNSW=====================" << std::endl;
    std::cout << "Database size: " << this->vectors_.size() << std::endl;
    for (int layerID = this->layers.size() - 1; layerID >= 0; layerID--)
    {
        const auto &layerGraph = this->layers[layerID];
        if (layerGraph.isEmpty())
            continue;
        std::cout << "layer " << layerID << ": ";
        layerGraph.summarize();
    }
}

void HNSW::save(const std::string &filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    cereal::BinaryOutputArchive archive(ofs);
    archive(*this);
}

HNSW loadHNSW(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    cereal::BinaryInputArchive archive(ifs);
    HNSW hnsw(1, 1, 1); // these init parameters will be replaced
    archive(hnsw);
    hnsw.prepareMemory();
    return hnsw;
}