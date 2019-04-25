#pragma once
#include <dune/hpdg/parallel/communicationhpdg.hh>
#include <dune/parmg/parallel/communicationp1.hh>
namespace Dune {
namespace HPDG {

  template<typename Basis>
  void updateDegrees(Basis& basis, ParMG::CommHPDG& comm) {
    std::vector<int> degrees(basis.size(),-1);
    auto lv = basis.localView();

    // copy degrees
    for(const auto& element : elements(basis.gridView())) {
      // only copy for owned degrees
      if(element.partitionType() != InteriorEntity)
        continue;
      lv.bind(element);
      degrees[lv.index(0)[0]] = basis.preBasis().degree(element);
      assert(degrees[lv.index(0)[0]] >0);
    }

    auto copy = ParMG::makeCopy<decltype(degrees)>(static_cast<ParMG::Comm&>(comm));
    // copy from master to the non masters
    copy(degrees);

    // reset degrees
    for(const auto& element : elements(basis.gridView())) {
      // the owned degrees should not be relevant
      if(element.partitionType() == InteriorEntity)
        continue;

      lv.bind(element);
      basis.preBasis().degree(element)= degrees[lv.index(0)[0]];
      //assert(degrees[lv.index(0)[0]] >0 && "later stage");
      if (degrees[lv.index(0)[0]] <=0)
        std::cout << "found a bad one\n";
    }
  }
}
}
