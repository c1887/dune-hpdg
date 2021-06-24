#pragma once

#include <memory>
#include <vector>
#include <optional>

#include <dune/hpdg/iterationsteps/dynamicblockgs.hh>
#include <dune/hpdg/iterationsteps/mg/multigrid.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>

#include <dune/hpdg/transferoperators/dynamicblocktransfer.hh>
#include <dune/hpdg/transferoperators/dynamicordertransfer.hh>
#include <dune/hpdg/transferoperators/ordertransfer.hh>

#include <dune/hpdg/assemblers/dgtodggridtransferassembler.hh>
#include <dune/solvers/common/defaultbitvector.hh>

// extra includes for P1 multigrid
#include <dune/common/fvector.hh>
#include <dune/common/power.hh>
#include <dune/fufem/assemblers/transferoperatorassembler.hh>
#include <dune/hpdg/transferoperators/blocktransfer.hh>
#include <dune/hpdg/assemblers/dgtocgtransferassembler.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/solvers/iterationsteps/blockgssteps.hh>
#include <dune/solvers/iterationsteps/multigridstep.hh>
#include <dune/solvers/transferoperators/compressedmultigridtransfer.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>

namespace Dune {
namespace HPDG {
namespace MultigridSetup {

template<typename Grid>
class P1Multigrid
{
public:
  using Matrix = BCRSMatrix<FieldMatrix<double, 1, 1>>;
  using Vector = BlockVector<FieldVector<double, 1>>;
  using BitVector = Solvers::DefaultBitVector_t<Vector>;
  using MG = MultigridStep<Matrix, Vector, BitVector>;
  using TransferOp = CompressedMultigridTransfer<Vector, BitVector, Matrix>;

  P1Multigrid(const Grid& grid, int smoothingSteps = 3)
  {
    auto assembler = TransferOperatorAssembler<Grid>(grid);
    // we dont need to store this vector all the time, as the transfer operators
    // will get a copy anyway.
    auto transferMats = std::vector<std::shared_ptr<Matrix>>();
    assembler.assembleMatrixHierarchy(transferMats);

    transfers_.resize(transferMats.size());
    for (size_t i = 0; i < transfers_.size(); ++i) {
      transfers_[i].setMatrix(transferMats[i]);
    }
    auto smoother = Solvers::BlockGSStepFactory<Matrix, Vector>::create(
      Solvers::BlockGS::LocalSolvers::gs());

    step_ = MG();
    step_.setTransferOperators(transfers_);
    step_.setSmoother(std::move(smoother));
    step_.setMGType(1, smoothingSteps, smoothingSteps);
  }

  auto& multigrid() { return step_; }

private:
  std::vector<TransferOp> transfers_;
  MG step_;
};

/**
 * Idea: have a coarse grid solver which is just regular
 * geometric multigrid for continuous P1.
 * If the grid is noncorming, we just dont mind.
 *
 * This has to do two things:
 * - restrict to CG (incl. using "classic" data types such as BlockVector and
 * BCRSMatrix)
 * - perform multigrid step (dune-solvers)
 */
template<typename Matrix, typename Vector, typename Grid, typename Smoother>
class P1MultigridCoarseSolver
{
  using P1Matrix = typename P1Multigrid<Grid>::Matrix;
  using P1Vector = typename P1Multigrid<Grid>::Vector;
  using P1BitVector = typename P1Multigrid<Grid>::BitVector;
  using BV =
     BlockVector<FieldVector<double, Dune::power(2, (int) Grid::dimension)>>;

public:
  P1MultigridCoarseSolver(std::shared_ptr<Matrix> fine,
                          const Grid& grid,
                          const Smoother& smoother)
    : fine_(fine)
    , p1MG_(grid)
    , smoother_(smoother)
  {
    dgToCg_ = DGtoCG();

    auto cbasis = Functions::LagrangeBasis<typename Grid::LeafGridView, 1>(
      grid.leafGridView());
    auto basis =
      Functions::DynamicDGQkGLBlockBasis<typename Grid::LeafGridView>(
        grid.leafGridView(), 1);

    dgToCg_.setup([&](auto& matrix) {
      Dune::HPDG::assembleDGtoCGTransferOperator(matrix, cbasis, basis);
    });
  }

  static auto dynamicBVtoBV(const Vector& source)
  {
    auto ret = BV(source.N());
    for (size_t i = 0; i < source.N(); ++i) {
      DUNE_ASSERT_BOUNDS(source[i].N() == ret[i].N());
      for (size_t j = 0; j < ret[i].N(); ++j) {
        ret[i][j] = source[i][j];
      }
    }
    return ret;
  }

  void operator()(Vector& x, const Vector& b)
  {
    // step 0: presmooth
    // TODO: maybe let the number iterations not be fixed.
    auto ignore = Solvers::DefaultBitVector_t<Vector>(x.size());
    for (size_t i = 0; i < ignore.size(); ++i) {
      ignore[i].resize(x[i].size());
      ignore[i].unsetAll();
    }
    smoother_.setIgnore(ignore);
    smoother_.preprocess();

    smoother_.setProblem(*fine_, x, b);
    for (int i = 0; i < 3; ++i)
      smoother_.iterate();

    auto r = b;
    auto c = x;
    c = 0;
    fine_->mmv(x, r);

    // step 1:
    auto x_cg = P1Vector();
    auto b_cg = P1Vector();
    // restrict c and r to CG space.
    auto cc = dynamicBVtoBV(c);
    dgToCg_.restrict(cc, x_cg);
    auto rr = dynamicBVtoBV(r);
    dgToCg_.restrict(rr, b_cg);

    // restrict matrix such that we have a CG matrix.
    P1Matrix cg_fine_;
    dgToCg_.galerkinRestrictSetOccupation(*fine_, cg_fine_);
    dgToCg_.galerkinRestrict(*fine_, cg_fine_);

    // step 2:
    // prepare MG.
    auto p1ignore = P1BitVector(x_cg.size(), false);
    p1MG_.multigrid().setProblem(cg_fine_, x_cg, b_cg);
    p1MG_.multigrid().setIgnore(p1ignore);

    // step 3: let MG run
    p1MG_.multigrid().preprocess();
    p1MG_.multigrid().iterate();

    // step 4:
    // TODO :project back to DG-P1
    cc = 0;
    dgToCg_.prolong(x_cg, cc);

    // step 4.5: add correction
    // x += cc;
    // okay, do it by hand for now.
    for (size_t i = 0; i < x.size(); ++i) {
      for (size_t j = 0; j < x[i].size(); ++j) {
        x[i][j] += cc[i][j];
      }
    }

    // step 5: postsmooth
    for (int i = 0; i < 3; ++i)
      smoother_.iterate();
  }

private:
  std::shared_ptr<Matrix> fine_;
  P1Multigrid<Grid> p1MG_;
  Smoother smoother_;
  using DGtoCG =
    HPDG::BlockTransferOperator<BV, 1, Solvers::DefaultBitVector_t<BV>, Matrix>;
  DGtoCG dgToCg_;
};
template<typename MatrixType,
         typename Vector,
         int dim,
         typename Grid,
         typename GS = HPDG::Imp::GSCore>
struct MultigridData_DGCG
{

  using Matrix = MatrixType;
  using PTransfer = Dune::HPDG::DGOrderTransfer<Vector, dim>;
  using GridTransfer = DynamicBlockTransferOperator<Vector>;

  using LocalSolver = GS;

  using Smoother = Dune::HPDG::
    DynamicBlockGS<Matrix, Vector, Solvers::DefaultBitVector_t<Vector>, GS>;
  using CoarseSolver = P1MultigridCoarseSolver<Matrix, Vector, Grid, Smoother>;

  // system matrices on each level
  std::vector<std::shared_ptr<Matrix>> systemMatrix;

  // transfer operators that restrict only polynomial degrees
  std::vector<std::shared_ptr<PTransfer>> pTransfer;

  std::optional<CoarseSolver> coarseSolver;
  // transfer operators that transfer between grid levels (for piecewise linear
  // DG!) we dont use these here since we do the geometric MG in the coarse
  // solver std::vector<std::shared_ptr<GridTransfer>> gridTransfer;
};

namespace Impl {
template<typename Matrix, typename Vector, int dim, typename Grid, typename GS>
class TransferOperators_DGCG
{
public:
  TransferOperators_DGCG(MultigridData_DGCG<Matrix, Vector, dim, Grid, GS>* data)
    : data_(data)
  {}

  void restrictMatrix(std::size_t coarseLevel,
                      const Matrix& fine,
                      Matrix& coarse)
  {
    data_->pTransfer[coarseLevel]->galerkinRestrict(fine, coarse);
  }

private:
  MultigridData_DGCG<Matrix, Vector, dim, Grid, GS>* data_;
};
}
/** Prepare multigrid data by resizing properly and
 * setting up transfer operators.
 *
 *
 * TODO: we dont really use the grid here. We only need the dim.
 */
template<typename Vector,
         typename Grid,
         typename M,
         typename LocalSolver = HPDG::Imp::GSCore>
auto
setupData_DGCG(const Grid& grid, int maximalDegree, M& matrix)
{
  constexpr const int dim = Grid::dimension;
  using MGData = MultigridData_DGCG<M, Vector, dim, Grid, LocalSolver>;
  MGData mgData;

  const auto pLevels = static_cast<int>(
    std::log2(maximalDegree)); // number of transfers for p levels
  const auto levels = pLevels;

  mgData.systemMatrix.resize(levels + 1);
  for (auto& matrix : mgData.systemMatrix)
    matrix = std::make_shared<typename MGData::Matrix>();
  mgData.systemMatrix.back() = Dune::stackobject_to_shared_ptr(matrix);

  mgData.pTransfer.resize(pLevels); // will be initialized in the next for loop

  // setup p-transfer
  for (int l = levels - 1; l >= 0; --l) {
    int currentIdx = l;
    int halfedOrder = maximalDegree / ((pLevels - currentIdx) * 2);
    auto& p = mgData.pTransfer[currentIdx];
    p = std::make_shared<typename MGData::PTransfer>();
    p->setup(*mgData.systemMatrix[l + 1], halfedOrder); // setup up transfer
    p->galerkinRestrictSetOccupation(*mgData.systemMatrix[l + 1],
                                     *mgData.systemMatrix[l]);
  }


  using Smoother = Dune::HPDG::
    DynamicBlockGS<M, Vector, Solvers::DefaultBitVector_t<Vector>, LocalSolver>;
  mgData.coarseSolver = typename MGData::CoarseSolver(mgData.systemMatrix.front(), grid, Smoother());

  return mgData;
}

template<typename Matrix, typename Vector, int dim, typename Grid, typename GS>
auto
setupLevelOps_DGCG(const MultigridData_DGCG<Matrix, Vector, dim, Grid, GS>& data)
{
  using LevelOps = typename Multigrid<Vector>::LevelOperations;
  std::vector<LevelOps> ops(data.systemMatrix.size()); // size levels+1

  ops.back().apply = operatorFromMatrix<Vector>(data.systemMatrix.back());

  // setup for p transfers
  auto& matrices = data.systemMatrix;
  auto& ptransfer = data.pTransfer;

  for (int l = ops.size() - 2; l >= 0; --l) {
    auto currentIdx = l;

    ops[l].apply = operatorFromMatrix<Vector>(matrices[l]);

    ops[l + 1].restrict =
      restrictFromMultigridTransfer<Vector>(ptransfer[currentIdx]);
    ops[l + 1].prolong =
      prolongFromMultigridTransfer<Vector>(ptransfer[currentIdx]);
  }

  // setup smoothers
  for (std::size_t l = 0; l < ops.size(); ++l) {
    using Smoother = Dune::HPDG::
      DynamicBlockGS<Matrix, Vector, Solvers::DefaultBitVector_t<Vector>, GS>;
    auto smoother = std::make_shared<Smoother>(); // no memory leak because
                                                  // shared_ptrs will be copied
    smoother->setMatrix(*matrices[l]);
    ops[l].preSmoother = ops[l].postSmoother =
      smootherFromIterationStep2<Vector>(smoother);
  }

  return ops;
}

template<typename M, typename Vector, int dim, typename Grid, typename GS>
void
renewMatrixHierachy(MultigridData_DGCG<M, Vector, dim, Grid, GS>& data)
{
  auto transfer = Impl::TransferOperators_DGCG<M, Vector, dim, Grid, GS>(&data);

  for (int l = data.systemMatrix.size() - 2; l >= 0; --l) {
    transfer.restrictMatrix(
      l, *data.systemMatrix[l + 1], *data.systemMatrix[l]);
  }
}

template<typename Matrix, typename Vector, int dim, typename Grid, typename GS>
class DGMultigridStep_DGCG
  : public LinearIterationStep<Matrix,
                               Vector,
                               Solvers::DefaultBitVector_t<Vector>>
{

public:
  auto& multigridData() { return data_; }

  auto& multigrid() { return multigrid_; }

  virtual void iterate()
  {
    auto b =
      *(this->rhs_); // the MG impl. modifies the rhs, hence we make a copy here
    auto& x = *(this->x_);

    multigrid_.apply(x, b);
  }

  virtual void preprocess()
  {
    // update fine matrix
    data_.systemMatrix.back() = std::const_pointer_cast<Matrix>(this->mat_);

    /* if fine matrix was changed, the `apply` part of the level operations
     * on the finest level needs to be updated. Unfortunately, the level ops are
     * not exposed, therefore we have to reset the all of them via this setup
     * tool:
     */
    multigrid_.levelOperations(setupLevelOps_DGCG(data_));

    // recalculate matrix hierarchy
    HPDG::MultigridSetup::renewMatrixHierachy(data_);
  }

private:
  MultigridData_DGCG<Matrix, Vector, dim, Grid, GS> data_;
  Multigrid<Vector> multigrid_;
};

template<typename Vector,
         typename GS = HPDG::Imp::GSCore,
         typename Grid,
         typename Matrix>
auto
multigridSolver_DGCG(const Grid& grid,
                     int maximalDegree,
                     const Matrix& matrix,
                     int coarseIterations = 5)
{
  constexpr const int dim = Grid::dimension;
  auto mg = std::make_shared<DGMultigridStep_DGCG<Matrix, Vector, dim, Grid, GS>>();

  Matrix& mut_matrix = const_cast<Matrix&>(matrix);
  mg->multigridData() =
    setupData_DGCG<Vector, Grid, Matrix, GS>(grid, maximalDegree, mut_matrix);
  mg->multigrid().levelOperations(setupLevelOps_DGCG(mg->multigridData()));

  // GS coarse solver

  // TODO: Hier das geometrische P1 Mehrgitter ankleben.
  // auto coarse_matrix_ptr = mg->multigridData().systemMatrix.front();
  // auto coarse_solver = [coarse_matrix_ptr, coarseIterations](auto& x,
  //                                                            const auto& b) {
  //   using Smoother = Dune::HPDG::
  //     DynamicBlockGS<Matrix, Vector, Solvers::DefaultBitVector_t<Vector>, GS>;
  //   auto smoother = std::make_shared<Smoother>();
  //   smoother->setProblem(*coarse_matrix_ptr, x, b);
  //   for (int i = 0; i < coarseIterations; i++)
  //     smoother->iterate();
  // };
  // mg->multigrid().coarseSolver(coarse_solver);
  mg->multigrid().coarseSolver(*(mg->multigridData().coarseSolver));

  return mg;
}

}
}
}
