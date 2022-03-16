#include <config.h>
#include <dune/common/filledarray.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/timer.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/hpdg/buildingblocks/matrices.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/test/testobjects.hh>

using namespace Dune;

int
main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  constexpr const int dim = 2;
  using Grid = YaspGrid<dim>;
  auto grid = Grid({ { 1.0, 1.0 } }, { { 5, 5 } });

  using GV = Grid::LeafGridView;

  TestSuite suite;

  constexpr const double penalty = 1.6;

  constexpr const auto dirichlet_values = std::array<bool, 2>{ true, false };

  for (auto dirichlet : dirichlet_values) {
    for (int order = 1; order < 4; ++order) {
      auto basis = Dune::Functions::DynamicDGQkGLBlockBasis<GV>(
        grid.leafGridView(), order);

      // Laplace
      {
        auto reference =
          dynamicStiffnessMatrix(grid, order, penalty, dirichlet);
        auto bb_mat =
          HPDG::BuildingBlocks::laplace<GV>(basis, penalty, dirichlet);

        reference -= bb_mat;
        suite.check(reference.frobenius_norm() < 1e-13)
          << "Error for Laplace is " << reference.frobenius_norm();
      }

      // Mass
      {
        auto reference =
          dynamicMassMatrix(grid, order);
        auto bb_mat =
          HPDG::BuildingBlocks::mass<GV>(basis);

        reference -= bb_mat;
        suite.check(reference.frobenius_norm() < 1e-13)
          << "Error for mass is " << reference.frobenius_norm();
      }
    }
  }

  return suite.exit();
}
