#pragma once
#include "common.h"
#include <vector>
#include <stdint.h>
#include <optional>
#include "heap.h"
#include "visited.h"
#include <cereal/access.hpp>

#define NSW_LAYERS_PREALLOC 21
#define DIST_FUNC squaredEuclideanDistance

void enableProfiling();
void disableProfiling();
void resetProfilingCounter();
uint64_t getCountDistCalc();

class NSW
{
  // contains 2 types of index: nodeID and nodePos
  // nodeID is the external ID given to this graph when a node is inserted
  // nodePos is the internal ID this graph gives a node when it is inserted
private:
  // max number of nodes
  uint64_t maxNumNodes_;
  // current number of nodes
  uint64_t numNodes_;
  // Neighbor list. Array of length maxNumNodes_
  // If a node does not exist, its location on this array is nullopt
  // A member on this array indicates a node's list of neighbors
  // Each element at index `pos` is a list of `nodePos` pointing to `pos`'s neighbors
  std::vector<std::optional<std::vector<nodeID_t>>> neighbors_;
  // nodePos of the default entrypoint for the search process
  nodeID_t defaultEntryID_;

public:
  // constructor
  NSW(uint64_t maxNumNodes = 10000);
  // get entrypoint nodeID when searching in this graph
  nodeID_t getEntrypointID() const;
  // how many neighbors does a node have?
  uint32_t getNeighborhoodSize(nodeID_t nodeID);
  // given a nodeID, find its neighbors' nodeIDs
  const std::vector<nodeID_t> &getNeighborsId(nodeID_t nodeID) const;
  // check if this graph is empty
  bool isEmpty() const;
  // get number of nodes in this graph
  uint32_t getNumNodes() const;
  // check if this graph contains a nodeID
  bool containsNode(nodeID_t nodeID) const;
  // add a given nodeID to this graph and connect it to the given neighbors (nodeIDs)
  void addNode(nodeID_t nodeID, const std::vector<nodeID_t> &neighborsNodeIds);
  // remove connection between 2 nodes at nodePos pos1 and pos2
  void removeConnection(nodeID_t nodeID1, nodeID_t nodeID2);
  // find and set default entrypoint (point with highest degree)
  void findAndSetDefaultEntry();
  // print a summary to stdout
  void summarize() const;
  // print the nodes in this graph
  void printNodes() const;
  // serialization access for cereal
  template <class Archive>
  void serialize(Archive &ar)
  {
    ar(maxNumNodes_, numNodes_, neighbors_, defaultEntryID_);
  }
  // functions for serializing/de-serializing
  void prepareSerializing();
  void postDeserializing();
};

// square euclidean distance
float squaredEuclideanDistance(const std::vector<float> &v1, const std::vector<float> &v2);

// Assumptions:
// 1. Using squared Euclidean distance
// 2. 128-dim vectors
class HNSW
{
private:
  // max number of nodes
  uint64_t maxNumNodes_;
  // exponential scaling parameter for layer distribution
  float mL_;
  // number of neighbors to search for when inserting a node
  uint32_t efc_;
  // number of neighbors to attempt to connect when inserting a node
  uint32_t M_;
  // max number of neighbors a node can have at any time
  uint32_t Mmax0_; // at layer 0
  uint32_t Mmax_;  // at higher layers
  // database contains all vectors
  std::vector<std::vector<float>> vectors_;
  // number of layers (non-empty in the preallocated layers)
  uint32_t nLayers_;
  // reserved priority queue of candidates for searchLayer()
  DaryHeap<4, std::pair<float, nodeID_t>, std::greater<std::pair<float, nodeID_t>>> searchCandidates_;
  // reserved visited set for searchLayer()
  VisitedCheck searchVisited_;
  // reserved distances array for checking out neighbors on searchLayer()
  std::vector<float> searchDists_;
  // get a vector for node at nodeID
  const std::vector<float> &getVector_(nodeID_t nodeID) const;
  // get the top non-empty layer of NSW
  NSW &getTopLayer_();
  // given total number of nodes, divide this number into layers and return the number of nodes in each layer
  std::vector<uint32_t> calculateNumNodesInLayers_(uint32_t total) const;
  // calculate distance between 2 node IDs
  float calcDistBtw2Ids_(nodeID_t nodeID1, nodeID_t nodeID2) const;
  // calculate distance between a query and a nodeID
  float calcDistToId_(const std::vector<float> &query, nodeID_t nodeID) const;
  // find a vector in the database that is the closest to the centroid of the vectors
  // from start to end-1 (start is included but end is excluded)
  // return the ID of the vector (nodeID)
  nodeID_t findCentroid_(nodeID_t start, nodeID_t end) const;
  // select M points from candidates to connect to query
  // candidates: List of (distance_to_query, nodeID)
  // Return: List of selected (distance_to_query, nodeID) from candidates, modified in-place
  // assume candidates are already sorted by distance
  void selectNeighborsSimple_(
      const std::vector<float> &query, std::vector<search_result_t> &candidates, uint32_t M) const;
  // select M points from candidates to connect to query
  // Return: List of selected (distance_to_query, nodeID) from candidates, modified in-place
  // assume candidates are already sorted by distance
  void selectNeighborsHeuristic_(
      const std::vector<float> &query, std::vector<search_result_t> &candidates, uint32_t M, uint32_t layerID);
  // insert a vector with ID=nodeID into layer layerID (and layers below it).
  void insertLayer_(uint32_t layerID, const std::vector<float> &vector, nodeID_t nodeID);

public:
  // Constructor
  HNSW(uint32_t dim, uint32_t efc, uint32_t M, uint64_t maxNumNodes = 10000);
  // NSW layers
  std::vector<NSW> layers;
  // search for a query's nearest neighbors within an NSW layer
  // entrypointID: nodeID on the vectors array. set to -1 to use layer"s default entry point
  // ef: limit size of candidate list during search
  // Return: list of ef tuples (distance, nodeID) ordered increasingly by distance
  std::vector<search_result_t> searchLayer(uint32_t layerID, nodeID_t entrypointID, const std::vector<float> &query, uint32_t ef);
  // search for a query's nearest neighbors in the entire database
  // Return: list of ef tuples (distance, nodeID) ordered increasingly by distance
  std::vector<search_result_t> search(const std::vector<float> &query, uint32_t ef);
  // parallel search for multiple queries (2d array, first dim is batch_size, 2nd dim is vector size)
  // Return: list of (list of ef tuples (distance, nodeID) ordered increasingly by distance)
  std::vector<std::vector<search_result_t>> searchParallel(
      const std::vector<std::vector<float>> &queries, uint32_t ef);
  // get number of layers
  uint32_t getNumLayers();
  // build the index from a list of vectors
  void buildIndex(const std::vector<std::vector<float>> &vectors);
  // print a summary to stdout
  void summarize() const;
  // serialization access for cereal
  template <class Archive>
  void serialize(Archive &ar)
  {
    ar(maxNumNodes_, mL_, efc_, M_, Mmax0_, Mmax_, vectors_, nLayers_, layers);
  }
  // saving and loading index
  void save(const std::string &filename);
  // initialize intermediate data
  void prepareMemory();
};

HNSW loadHNSW(const std::string &filename);