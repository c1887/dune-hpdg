#include <stdexcept>

#include <dune/common/power.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/indices.hh>

#include <dune/istl/bvector.hh>

#include <dune/hpdg/common/arraytotuple.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslobatto.hh>
#include <dune/hpdg/transferoperators/dgmultigridtransfermatrixfree.hh>

#include <dune/functions/common/utility.hh>

namespace Dune {
namespace HPDG {
  /** \brief Generate a tuple of transfer operators for DG-p-multilevel
   *
   * @param levels tuple of std::pairs of Dune::Indices
   * @param grid The given grid
   *
   * Currently, this uses the matrix free impl. of the transfer. This may change.
   */
  template<class GridType, class T>
  auto make_dgTransferTuple(const T& levels, const GridType& grid) {

    // Check if input is valid
    namespace H = Dune::Hybrid;
    H::forEach(H::integralRange(H::size(levels)), [&](auto i)
    {
      auto& pair = H::elementAt(levels, i);

      // pairs have to be ordered in descending order
      if (pair.first.value < pair.second.value) {
        DUNE_THROW(Dune::Exception, 
            "Levels have to be in descending order (pair " <<
            i << " is (_" << pair.first.value <<", _"<< pair.second.value<<") but should be (_"
            << pair.second.value << ", _" << pair.first.value <<")" );
      }

      // the levels have to be consecutive in the sense that the first element of a pair has to be the same 
      // as the second element of the following pair
      const auto n = H::size(levels);
      const auto lastIndex = Dune::index_constant<n-1>();
      H::ifElse(H::equals(i, lastIndex), [&](auto){}, [&] (auto id) {
        const auto ii = id(i);
        const auto nextIndex = Dune::index_constant<ii+1>();
        auto& nextpair = id(H::elementAt(levels, id(nextIndex)));
        if (pair.first.value != nextpair.second.value) 
          DUNE_THROW(Dune::Exception, "Fine level of pair " << ii.value << " does not match coarse level of pair " << nextIndex.value <<"!");
          });
    });

    // lambda that acts on a std::pair
    auto buildTransferFromPair = [&] (auto levelPair) {
      constexpr auto dim = GridType::dimensionworld;

      constexpr auto pfine = levelPair.first.value; // fine level
      constexpr auto pcoarse = levelPair.second.value; // coarse level

      const int rows = Dune::StaticPower<pfine+1, dim>::power;
      const int cols = Dune::StaticPower<pcoarse+1, dim>::power;

      using FV = Dune::BlockVector<Dune::FieldVector<double, rows>>;
      using DGTransfer = Dune::HPDG::DGMultigridTransferMatrixFree<FV, cols>;
      auto dgTransfer = DGTransfer();

      auto fineFE = Dune::QkGaussLobattoLocalFiniteElement<typename GridType::LeafGridView::ctype, double, dim, pfine>();
      auto coarseFE = Dune::QkGaussLobattoLocalFiniteElement<typename GridType::LeafGridView::ctype, double, dim, pcoarse>();

      // compute transfer from fine to coarse FE
      dgTransfer.setup(coarseFE, fineFE);
      return dgTransfer;
    };

    return Dune::Functions::transformTuple(buildTransferFromPair, levels);
  }
}
}
