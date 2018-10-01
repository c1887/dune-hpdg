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

    auto fullSum = std::accumulate(std::begin(vec_sorted), std::end(vec_sorted), K(0));

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

  /** \brief Marks an element e for refinement and gurantees that the level difference
   * between two neighboring elements is not greater than a given value.
   *
   * \warning Note that this might have bad complexity, I never checked.
   */
  template<class G, class E>
  void marker(G& grid, const E& e, int max_diff) {

    // mark the element e
    grid.mark(1, e);

    // check if neighbors are too coarse, i.e. there level difference is too big. If so, refine them too
    auto level = e.level()+1; // +1 from the refinement above
    for (auto& is : intersections(grid.leafGridView(), e)) {
      if (!is.neighbor())
        continue;
      auto outside = is.outside();
      auto outsidelevel = outside.level() + grid.getMark(outside);
      if (level - outsidelevel > max_diff) {
        if (not grid.getMark(outside))
          marker(grid, outside, max_diff);
      }
    }
  }

}} // end namespace Dune::HPDG

#endif//DUNE_HPDG_ESTIMATORS_UTILITY_HH
