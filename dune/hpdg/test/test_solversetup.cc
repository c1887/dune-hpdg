#if HAVE_CONFIG
#include "config.h"
#endif

#include <dune/common/parallel/mpihelper.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/hpdg/iterationsteps/mg/multigrid.hh>
#include <dune/hpdg/iterationsteps/mg/mgwrapper.hh>
#include <dune/hpdg/iterationsteps/solversetup.hh>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/common/dynamicbvector_specializations.hh>
#include <dune/hpdg/test/testobjects.hh>

#include <dune/solvers/norms/energynorm.hh>
#include <dune/solvers/solvers/loopsolver.hh>

// TODO: make this thing actually test things (besides being able to compile)
int main(int argc, char** argv) {
  using namespace Dune;
  Dune::MPIHelper::instance(argc, argv);

  using Vector = HPDG::DynamicBlockVector<FieldVector<double, 1>>;

  constexpr const int dim = 2;
  YaspGrid<dim> grid({1.,1.}, {2,2});
  grid.globalRefine(2);

  int order = 2;
  auto matrix = dynamicStiffnessMatrix(grid, order);
  auto rhs = dynamicRightHandSide(grid, order);
  auto x = Vector::createZeroVector(rhs);

  using BitSetVector = Dune::Solvers::DefaultBitVector_t<Vector>;

  auto mg = HPDG::MultigridSetup::multigridSolver<Vector>(grid, order, matrix);
  mg->setProblem(matrix, x, rhs);

  auto norm = std::make_shared<Dune::Solvers::EnergyNorm<decltype(matrix), Vector>>(matrix);

  using Solver = Dune::Solvers::LoopSolver<Vector, BitSetVector>;

  int iter = 15;
  double tol = 1e-8;

  auto solver = Solver(mg, iter, tol, norm, Solver::FULL);

  solver.preprocess();
  solver.solve();

  return 0;
}
