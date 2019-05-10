#pragma once
#include <dune/hpdg/parallel/communicationhpdg.hh>
#include <dune/parmg/parallel/communicationp1.hh>
#include <dune/hpdg/common/dynamicbvector.hh>
namespace Dune {
namespace HPDG {

  // TODO: This looks still very messy.
  template<typename Basis>
  void updateDegrees(Basis& basis, ParMG::CommHPDG& comm) {
    using V = DynamicBlockVector<FieldVector<double,1>>; // for some reason, std::vector did not work on first try. TODO
    V degrees(basis.size());
    for(std::size_t i = 0; i < basis.size(); i++) {
      degrees.blockRows(i) = 1;
    }
    degrees.update();
    degrees = -1.;
    auto lv = basis.localView();

    // copy degrees
    for(const auto& element : elements(basis.gridView())) {
      // only copy for owned degrees
      if(element.partitionType() != InteriorEntity)
        continue;
      lv.bind(element);
      degrees[lv.index(0)[0]][0] = (double) basis.preBasis().degree(element);
      assert(degrees[lv.index(0)[0]][0] >0.);
    }

    auto copy = ParMG::makeDGCopy<decltype(degrees)>(comm);
    // copy from master to the non masters
    copy(degrees);

    // reset degrees
    for(const auto& element : elements(basis.gridView())) {
      // the owned degrees should not be relevant
      if(element.partitionType() == InteriorEntity)
        continue;

      lv.bind(element);
      assert(degrees[lv.index(0)[0]][0] >0.);
      basis.preBasis().degree(element)= (int) degrees[lv.index(0)[0]][0];
    }
  }
}
}
