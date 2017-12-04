#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/indices.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/hpdg/transferoperators/makedgtransfertuple.hh>
#include <dune/hpdg/iterationsteps/dgmultigridstep.hh>

#include <dune/solvers/iterationsteps/blockgssteps.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/norms/energynorm.hh>
#include <dune/solvers/solvers/umfpacksolver.hh>

#include "testobjects.hh"

using namespace Dune;

TestSuite test_DGMultigridStep() {
  TestSuite suite("test_DGMultigridStep");

  using namespace Dune::Indices;
  constexpr auto k = _4;
  const double tolerance = 1e-10; // max. allowed error
  /* Set up grid */
  using GridType = YaspGrid<2>;
  auto grid = StructuredGridFactory<GridType>::createCubeGrid({0,0}, {1.0,1.0}, {{4,4}});

  auto matrix = stiffnessMatrix<k.value>(*grid);
  using Matrix = std::decay_t<decltype(matrix)>;

  // set up some "rhs"
  using Vector = BlockVector<FieldVector<double, Dune::StaticPower<k.value+1, 2>::power> >;
  Vector b(matrix.N());
  b=1.;
  
  auto x=b;
  x=0.; // initial iterate

  // set up transfer operators for p-multigrid
  constexpr auto levelpairs = std::make_tuple(
      std::make_pair(_2, _1),
      std::make_pair(k, _2));
  auto transfer = HPDG::make_dgTransferTuple(levelpairs, *grid);

  // inner solver in block Gauss Seidel
  auto localSolver = Solvers::BlockGS::LocalSolvers::gs();

  // setup multigrid step
  auto multigridStep = Solvers::DGMultigridStep<Matrix, Vector, decltype(transfer), decltype(localSolver)>(matrix, x, b, transfer, localSolver);

#ifdef HAVE_SUITESPARSE_UMFPACK
  // setup UMFPACK base solver
  constexpr auto coarse_k = std::get<0>(levelpairs).second.value; // coarsest level
  using CV = BlockVector<FieldVector<double, Dune::StaticPower<coarse_k+1,2>::power> >; // vector class on coarse level
  using CM = BCRSMatrix<FieldMatrix<double, Dune::StaticPower<coarse_k+1,2>::power, Dune::StaticPower<coarse_k+1,2>::power> >; // matrix class on coarse level
  auto basesolver = Solvers::UMFPackSolver<CM, CV>();
  multigridStep.setBaseSolver(basesolver);
#endif

  auto norm = EnergyNorm<Matrix, Vector>{matrix};
  auto solver = ::LoopSolver<Vector>{multigridStep, 99, 1e-13, norm, Solver::FULL};

  solver.preprocess();
  solver.solve();

  auto Ax = x;
  matrix.mv(x, Ax);
  suite.check(norm.diff(Ax, b) < tolerance, "Check if linear p-MG converged") << " Error is too great: " <<norm.diff(Ax, b);

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_DGMultigridStep());
  return suite.exit();
}
