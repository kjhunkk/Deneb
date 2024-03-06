#pragma once

#include <vector>
#include <memory>

#define DENEB_UTILITY_NAME utility_global_ptr
#define DENEB_UTILITY deneb::DENEB_UTILITY_NAME
#define DENEB_UTILITY_INITIALIZE() \
  DENEB_UTILITY = std::make_shared<deneb::Utility>()
#define DENEB_UTILITY_FINALIZE() DENEB_UTILITY.reset()

namespace deneb {
class Utility {
 public:
  Utility(){};
  ~Utility(){};

  void ComputeError(
      const std::string& filename, const double* solution, const double* exact_solution);
};
extern std::shared_ptr<Utility> DENEB_UTILITY_NAME;
}  // namespace deneb