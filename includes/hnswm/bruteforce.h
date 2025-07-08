#include <vector>
#include <string>
#include "common.h"
#include <cereal/access.hpp>

class BruteForce
{
private:
  // database contains all vectors
  std::vector<std::vector<float>> vectors_;

public:
  // Constructor
  BruteForce(uint32_t dim, uint64_t maxNumNodes = 10000);
  std::vector<search_result_t> search(const std::vector<float> &query, uint32_t k = 1);
  // build the index from a list of vectors
  void buildIndex(const std::vector<std::vector<float>> &vectors);

  // Replace boost serialization with cereal:
  template <class Archive>
  void serialize(Archive &ar)
  {
    ar(vectors_);
  }

  // saving and loading index
  void save(const std::string &filename);
};

BruteForce loadBruteForce(const std::string &filename);
