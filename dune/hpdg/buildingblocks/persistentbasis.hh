#ifndef DUNE_HDPG_BUILDINGBLOCKS_PERSISTENT_BASIS_HH
#define DUNE_HDPG_BUILDINGBLOCKS_PERSISTENT_BASIS_HH
#include <dune/hpdg/common/resizehelper.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/functionspacebases/persistentgridview.hh>
#include <dune/hpdg/functionspacebases/persistentgridviewdatatransfer.hh>
#include <dune/hpdg/gridfunctions/discretepersistentbasisfunction.hh>
namespace Dune::HPDG::BuildingBlocks {
template<typename GridView>
struct SavedBasis
{
  using GV = Dune::Functions::Experimental::PersistentGridView<GridView>;
  using Basis = Dune::Functions::DynamicDGQkGLBlockBasis<GV>;

  SavedBasis(const Dune::Functions::DynamicDGQkGLBlockBasis<GridView> basis)
    : pbasis(GV(basis.gridView()))
  {
    for (const auto& element : elements(basis.gridView()))
      pbasis.preBasis().degree(element) = basis.preBasis().degree(element);
  }

  // GV pgv;
  Basis pbasis;
};

template<typename GridView>
auto
saveDegrees(const Functions::DynamicDGQkGLBlockBasis<GridView>& basis, std::shared_ptr<const Dune::Functions::Experimental::PersistentGridView<GridView>> pgv)
{
  auto orders = Dune::HPDG::PersistentGridViewDataTransfer<GridView, int>(pgv);

  for (const auto& element : elements(basis.gridView())) {
    int order = basis.preBasis().degree(element);
    orders[element] = order;
  }
  return orders;
}

template<typename SavedDegrees, typename GV>
void
updateDegrees(const SavedDegrees& saved,
              Dune::Functions::DynamicDGQkGLBlockBasis<GV>& basis)
{
  for (const auto& element : elements(basis.gridView())) {
    basis.preBasis().degree(element) = saved[element];
  }
}

/** Compute coefficients of a function living on the saved basis in the
 * new, refined basis.
 *
 * @param saved Old state of the basis.
 * @param refinedBasis new, refined basis.
 * @param coeffs Coefficients of grid function wrt. the old basis.
 *
 * @returns Coefficients wrt. the new basis.
 */
template<typename OldGV, typename NewGV, typename Coeffs>
auto
interpolateIntoRefinedBasis(
  const SavedBasis<OldGV>& saved,
  const Dune::Functions::DynamicDGQkGLBlockBasis<NewGV>& refinedBasis,
  const Coeffs& coeffs)
{
  auto ret = Coeffs();
  Dune::HPDG::resizeFromBasis(ret, refinedBasis);

  auto coarse = Dune::Functions::makeDiscretePersistentBasisFunction<double>(
    saved.pbasis, coeffs);
  Dune::Functions::interpolate(refinedBasis, ret, coarse);
  return ret;
}
}
#endif // DUNE_HDPG_BUILDINGBLOCKS_PERSISTENT_BASIS_HH
