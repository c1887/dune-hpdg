#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/indices.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/hpdg/assemblers/dgtodggridtransferassembler.hh>

#include <dune/solvers/iterationsteps/blockgssteps.hh>
#include <dune/solvers/iterationsteps/multigridstep.hh>
#include <dune/solvers/transferoperators/densemultigridtransfer.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/norms/energynorm.hh>
#include <dune/solvers/solvers/umfpacksolver.hh>

#include "testobjects.hh"

using namespace Dune;

/* test geometric multigrid from dune-solvers with blocked DG 
 * matrices
 */
TestSuite test_geometricDGMultigrid() {
  TestSuite suite("test_geometricDGMultigrid");

  const double tolerance = 1e-10; // max. allowed error
  const int number_of_coarse_grids = 2;
  /* Set up grid */
  using GridType = YaspGrid<2>;
  auto grid = StructuredGridFactory<GridType>::createCubeGrid({0,0}, {1.0,1.0}, {{2,2}});
  grid->globalRefine(number_of_coarse_grids);


  auto matrix = stiffnessMatrix<1>(*grid); // Q1 DG
  using Matrix = std::decay_t<decltype(matrix)>;

  // set up some "rhs"
  auto b = rightHandSide<1>(*grid);
  using Vector=decltype(b);
  
  using BitVector = Solvers::DefaultBitVector_t<Vector>;
  auto x=b;
  x=0.; // initial iterate

  auto transfer = std::vector<std::shared_ptr<Matrix>>(number_of_coarse_grids);
  for (auto& t: transfer) t = std::make_shared<Matrix>();
  // compute transfer operator hierarchy (grid is allowed to be nonconforming!)
  HPDG::assembleDGGridTransferHierarchy(transfer, *grid);

  // copy into dune-solvers transfer operator
  using TransferOperator = DenseMultigridTransfer<Vector>;
  using TransferOperators = std::vector<std::shared_ptr<TransferOperator>>;
  TransferOperators transferOps(grid->maxLevel());
  for (size_t i = 0; i < transfer.size(); ++i)
  {
    // create transfer operator from level i to i+1
    transferOps[i] = std::make_shared<TransferOperator>();
    transferOps[i]->setMatrix(*transfer[i]);
  }

  // make some sanity checks
  {
    auto& topTransfer = transfer.back();
    suite.check(topTransfer->N() == x.size()) << "Transfer matrix has wrong number of rows";

    auto lastN = transfer.front()->M();
    for (const auto& t : transfer) {
      suite.check(t->N() >= t->M()) << "Transfer  operator has less rows than columns";
      suite.check(t->M() == lastN) << "Transfer operator's columns do now coincide with previous rows";
      lastN=t->N();
    }
  }

  // inner solver in block Gauss Seidel
  auto localSolver = Solvers::BlockGS::LocalSolvers::gs();
  // block Gauss Seidel
  auto smoother = Solvers::BlockGSStepFactory<Matrix, Vector, BitVector>::create(localSolver);

  // setup multigrid step
  auto multigridStep = Solvers::MultigridStep<Matrix, Vector, BitVector>(matrix, x, b);
  multigridStep.setSmoother(&smoother);
  multigridStep.setTransferOperators(transferOps);
  multigridStep.setMGType(1,5,5);

  // dune-solvers multigrid needs some ignore nodes
  BitVector dummy(x.size());
  multigridStep.setIgnore(dummy);

  auto norm = EnergyNorm<Matrix, Vector>{matrix};
  auto solver = ::LoopSolver<Vector>{&multigridStep, 99, 1e-13, &norm, Solver::FULL};

  solver.preprocess();
  solver.solve();

  auto Ax = x;
  matrix.mv(x, Ax);
  suite.check(norm.diff(Ax, b) < tolerance, "Check if linear geometric-MG converged") << " Error is too great: " <<norm.diff(Ax, b);

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_geometricDGMultigrid());
  return suite.exit();
}
