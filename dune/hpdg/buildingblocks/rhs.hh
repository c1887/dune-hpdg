#ifndef DUNE_HPDG_BUILDING_BLOCKS_RHS
#define DUNE_HPDG_BUILDING_BLOCKS_RHS
#include <dune/fufem/assemblers/dunefunctionsfunctionalassembler.hh>
#include <dune/fufem/assemblers/localassemblers/dunefunctionsl2functionalassembler.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

namespace Dune::HPDG::BuildingBlocks {

/**
 * @brief Create a vector representing the functional <g, .> on a space given by
 * its basis.
 *
 * @tparam Basis
 * @tparam G
 * @param basis
 * @param g
 * @return auto
 */
template<typename Basis, typename G>
auto
l2Functional(const Basis& basis, const G& g, const QuadratureRuleKey& key)
{
  using Vector = Dune::HPDG::DynamicBlockVector<Dune::FieldVector<double, 1>>;
  auto rhs = Vector();
  Dune::HPDG::resizeFromBasis(rhs, basis);

  auto rhsBE = Dune::Functions::istlVectorBackend(rhs);

  using FiniteElement =
    std::decay_t<decltype(basis.localView().tree().finiteElement())>;

  // assemble standard function \int fv
  {
    using GridType = typename Basis::GridView::Grid;
    auto rhsLocalAssembler = Dune::Fufem::
      DuneFunctionsL2FunctionalAssembler<GridType, FiniteElement, G>{ g, key };
    const auto localRHSlambda =
      [&](const auto& element, auto& localV, const auto& localView) {
        rhsLocalAssembler.assemble(
          element, localV, localView.tree().finiteElement());
      };
    Dune::Fufem::DuneFunctionsFunctionalAssembler<Basis> rhsAssembler(basis);
    // We only have the correct vector size, hence we only need to assemble the
    // Entries
    rhsAssembler.assembleBulkEntries(rhsBE, localRHSlambda);
  }

  return rhs;
}
}
#endif // DUNE_HPDG_BUILDING_BLOCKS_RHS