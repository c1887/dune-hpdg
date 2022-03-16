#ifndef DUNE_HPDG_BUILDING_BLOCKS_RHS
#define DUNE_HPDG_BUILDING_BLOCKS_RHS
#include <dune/fufem/assemblers/dunefunctionsboundaryfunctionalassembler.hh>
#include <dune/fufem/assemblers/dunefunctionsfunctionalassembler.hh>
#include <dune/fufem/assemblers/localassemblers/dunefunctionsl2functionalassembler.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>
#include <dune/hpdg/assemblers/localassemblers/ipdgboundaryassembler.hh>
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

/**
 * @brief Assembles the extra terms for the rhs when Dirichlet data is to be
 * incorporated.
 *
 * @tparam Basis
 * @tparam G
 * @param basis
 * @param g
 * @param key
 * @param penalty
 * @return auto
 */
template<typename Basis, typename G>
auto
dirichletData(const Basis& basis,
              const G& g,
              // const QuadratureRuleKey& key, // TODO: this is missing in the
              // idpg boundary assembler!
              double penalty)
{

  using Vector = Dune::HPDG::DynamicBlockVector<Dune::FieldVector<double, 1>>;
  using Grid = typename Basis::GridView::Grid;
  auto localView = basis.localView();

  auto dirichlet =
    Dune::Functions::makeAnalyticGridViewFunction(g, basis.gridView());

  auto r = Vector{};
  Dune::HPDG::resizeFromBasis(r, basis);
  r = 0.0;

  auto rBE = Dune::Functions::istlVectorBackend(r);
  HPDG::IPDGBoundaryAssembler<Grid, decltype(dirichlet)> ipdgBdrAssembler(
    dirichlet, true, true); // dirichlet data true, varying degree true
  ipdgBdrAssembler.setPenalty(penalty);
  const auto localRHSlambda =
    [&](const auto& bIt, auto& localV, const auto& localView) {
      ipdgBdrAssembler.assemble(bIt, localV, localView.tree().finiteElement());
    };
  BoundaryPatch<typename Basis::GridView> bpatch(basis.gridView(), true);
  auto bdrAssembler =
    Dune::Fufem::duneFunctionsBoundaryFunctionalAssembler(basis, bpatch);
  bdrAssembler.assembleBulk(rBE, localRHSlambda);
  assert(r.checkValidity());

  return r;
}
}
#endif // DUNE_HPDG_BUILDING_BLOCKS_RHS