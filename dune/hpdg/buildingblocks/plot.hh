#ifndef DUNE_HPDG_BUILDING_BLOCKS_PLOT_HH
#define DUNE_HPDG_BUILDING_BLOCKS_PLOT_HH

#include <dune/functions/functionspacebases/lagrangedgbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/hpdg/buildingblocks/details.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <string>

/** Building blocks for plotting DG functions.
 *
 * TODO: Maybe use dune-vtk for this.
 */
namespace Dune::HPDG::BuildingBlocks {

/** Plot a scalar function given by a grid function. */
template<typename GridFunction>
void
plot(const GridFunction& func,
     std::string filename,
     int refinementIntervals = 1)
{
  // TODO: This assumes that we can get the gridview via the basis() method.
  // Actually, this could be more general.
  auto vtkWriter =
    Dune::SubsamplingVTKWriter<typename GridFunction::GridView>(
      func.basis().gridView(), Dune::RefinementIntervals(refinementIntervals)
    );

  vtkWriter.addVertexData(
    func,
    Dune::VTK::FieldInfo("x", Dune::VTK::FieldInfo::Type::scalar, 1));
  vtkWriter.write(filename);
}

/** Plot a scalar function given by a basis and coefficients. */
template<typename GridView, typename Vector>
void
plot(const Functions::DynamicDGQkGLBlockBasis<GridView>& basis,
     const Vector& coeffs,
     std::string filename)
{
  const auto xFunction =
    Dune::Functions::makeDiscreteGlobalBasisFunction<double>(basis, coeffs);

  plot(xFunction, std::move(filename), Detail::maxDegree(basis));
}

/** Plots only the grid without any data attached. */
template<typename GridView>
void
plotGrid(const GridView& gv, std::string filename)
{
  auto vtkWriterbasic = Dune::VTKWriter<GridView>{ gv };
  vtkWriterbasic.write(filename);
}

template<typename GridView>
void
plotDiscretization(const Functions::DynamicDGQkGLBlockBasis<GridView>& basis,
                   std::string filename)
{
  auto degreeVector = [&]() {
    auto vec = BlockVector<FieldVector<double, 1>>(basis.size());
    auto lv = basis.localView();
    for (const auto& element : elements(basis.gridView())) {
      lv.bind(element);
      auto i = lv.index(0)[0];
      vec[i] = basis.preBasis().degree(element);
    }
    return vec;
  };
  const auto vec = degreeVector();

  const auto p0 =
    Dune::Functions::LagrangeDGBasis<GridView, 0>(basis.gridView());
  const auto pDistribution =
    Dune::Functions::makeDiscreteGlobalBasisFunction<double>(
      p0, Dune::Functions::istlVectorBackend(vec));
  auto vtk = Dune::VTKWriter<GridView>{ basis.gridView() };
  vtk.addCellData(
    pDistribution,
    Dune::VTK::FieldInfo("p", Dune::VTK::FieldInfo::Type::scalar, 1));
  vtk.write(filename);
}

} // end namespace Dune::HPDG::BuildingBlocks

#endif // DUNE_HPDG_BUILDING_BLOCKS_PLOT_HH
