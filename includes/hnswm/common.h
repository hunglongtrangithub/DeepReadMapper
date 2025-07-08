#include <stdint.h>
#include <utility>
#include <vector>

using nodeID_t = uint32_t;
using search_result_t = std::pair<float, nodeID_t>;
#define VECTOR_DIM 128

float squaredEuclideanDistance(const std::vector<float> &v1, const std::vector<float> &v2);
