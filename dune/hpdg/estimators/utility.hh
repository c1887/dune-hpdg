#ifndef DUNE_HPDG_ESTIMATORS_UTILITY_HH
#define DUNE_HPDG_ESTIMATORS_UTILITY_HH
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
namespace Dune {
namespace HPDG {
  /** \brief Returns the value at a given quantile of a vector 
   *
   *  This might be inaccurate if the vector has many equal elements.
   */
  template<typename K>
  K quantile(const std::vector<K>& vec, double quant) {
    auto vec_sorted = vec;
    std::sort(std::begin(vec_sorted), std::end(vec_sorted));

    // the following is not 100% correct, but it serves it's purpose for us
    auto quantile_value = vec_sorted[(size_t) (quant*vec_sorted.size())];

    return quantile_value;
  }

  /** \brief Returns a value v such that such the sum over all
   * elements >=v in vec will be at least a given fraction of the total sum.
   */
  template<typename K>
  K fraction(const std::vector<K>& vec, double frac) {
    auto vec_sorted = vec;
    std::sort(std::begin(vec_sorted), std::end(vec_sorted));

    auto fullSum = std::accumulate(std::begin(vec_sorted), std::end(vec_sorted), 0);

    auto thresh = frac*fullSum;
    
    if (thresh==0.0)
      return K(0.0);

    for(const auto& val : vec_sorted) {
      fullSum -= val;
      if (fullSum <= thresh)
        return val;
    }

    // logically, we should never end here
    // However, compiler doesn't end a function that does not end with a return
    assert(false);
    return K(0.0);
  }

}} // end namespace Dune::HPDG

#endif//DUNE_HPDG_ESTIMATORS_UTILITY_HH
