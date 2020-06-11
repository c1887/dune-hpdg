// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_COMMON_RESIZE_HELPER_HH
#define DUNE_HPDG_COMMON_RESIZE_HELPER_HH

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

namespace Dune {
namespace HPDG {
  template <class K, class B>
  void resizeFromBasis(DynamicBCRSMatrix<K>& matrix, const B& basis) {
    matrix.finishIdx();
    auto localView = basis.localView();
    auto prefix = typename B::SizePrefix {};
    for (const auto& e : elements(basis.gridView())) {
      localView.bind(e);
      prefix.push_back(localView.index(0)[0]);
      matrix.blockRows(prefix[0]) = basis.size(prefix);

      prefix.clear();
    }
    matrix.setSquare();

    // allocate memory
    matrix.update();
  }

  template <class K, class B>
  void resizeFromBasis(DynamicBlockVector<K>& v, const B& basis) {
    v.setSize(basis.size());
    auto localView = basis.localView();
    auto prefix = typename B::SizePrefix {};
    for (const auto& e : elements(basis.gridView())) {
      localView.bind(e);
      prefix.push_back(localView.index(0)[0]);
      // we're interested in the size of the next position, so we supply a prefix
      v.blockRows(prefix[0]) = basis.size(prefix);

      prefix.clear(); // this has to be done in every loop since we assume the current index is the first
    }

    // allocate memory
    v.update();
  }
}
}
#endif//DUNE_HPDG_COMMON_RESIZE_HELPER_HH
