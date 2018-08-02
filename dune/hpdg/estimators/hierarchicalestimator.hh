// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_ESTIMATOR_HIERARCHICALESTIMATOR_HH
#define DUNE_HPDG_ESTIMATOR_HIERARCHICALESTIMATOR_HH

#include <vector>

namespace Dune {
namespace HPDG {
  template<class Basis, class HierarchicalBasis>
  class HierarchicalEstimator {
    public:

    HierarchicalEstimator(const Basis& b, const HierarchicalBasis& hb) :
      basis_(b),
      hbasis_(hb) {}

    auto errors() const {
      return errors_;
    }

    /** \brief return the error value at a given quantile.
     *
     * Note that this might be inaccurate if many errors are
     * the same.
     */
    double getErrorThreshold(double quantile) const {
      using namespace std;

      auto errorsSorted = errors_;
      sort(begin(errorsSorted), end(errorsSorted));

      auto cutOff = static_cast<size_t>(ceil(errorsSorted.size() * quantile));

      return errorsSorted[cutOff];
    }

    protected:
      const Basis& basis_;
      const HierarchicalBasis& hbasis_;
      std::vector<double> errors_;
  };
} // end namespace HPDG
} // end namespace Dune
#endif
