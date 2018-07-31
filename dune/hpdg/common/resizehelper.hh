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
    for (const auto& e : elements(basis.gridView())) {
      localView.bind(e);
      matrix.blockRows(localView.index(0)[0]) = localView.size();
    }
    matrix.setSquare();
  }

  template <class K, class B>
  void resizeFromBasis(DynamicBlockVector<K>& v, const B& basis) {
    v.setSize(basis.size());
    auto localView = basis.localView();
    for (const auto& e : elements(basis.gridView())) {
      localView.bind(e);
      v.blockRows(localView.index(0)[0]) = localView.size();
    }
  }
}
}
#endif//DUNE_HPDG_COMMON_RESIZE_HELPER_HH
