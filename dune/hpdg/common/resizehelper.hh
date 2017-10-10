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
    auto localIS = basis.localIndexSet();
    for (const auto& e : elements(basis.gridView())) {
      localView.bind(e);
      localIS.bind(localView);
      matrix.blockRows(localIS.index(0)[0]) = localIS.size();
    }
    matrix.setSquare();
  }

  template <class K, class B>
  void resizeFromBasis(DynamicBlockVector<K>& v, const B& basis) {
    v.setSize(basis.size());
    auto localView = basis.localView();
    auto localIS = basis.localIndexSet();
    for (const auto& e : elements(basis.gridView())) {
      localView.bind(e);
      localIS.bind(localView);
      v.blockRows(localIS.index(0)[0]) = localIS.size();
    }
  }
}
}
#endif//DUNE_HPDG_COMMON_RESIZE_HELPER_HH
