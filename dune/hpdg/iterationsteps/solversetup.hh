#pragma once

#include <memory>

#include <dune/hpdg/iterationsteps/mg/multigrid.hh>
#include <dune/hpdg/iterationsteps/dynamicblockgs.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>

#include <dune/hpdg/transferoperators/ordertransfer.hh>
#include <dune/hpdg/transferoperators/dynamicordertransfer.hh>
#include <dune/hpdg/transferoperators/dynamicblocktransfer.hh>

#include <dune/hpdg/assemblers/dgtodggridtransferassembler.hh>
#include <dune/solvers/common/defaultbitvector.hh>


namespace Dune {
namespace HPDG {
namespace MultigridSetup {
  template<typename Vector, int dim>
  struct MultigridData {

    using Matrix = DynamicBCRSMatrix<FieldMatrix<double,1,1>>;
    using PTransfer = Dune::HPDG::DGOrderTransfer<Vector, dim>;
    using GridTransfer = DynamicBlockTransferOperator<Vector>;

    // system matrices on each level
    std::vector<std::shared_ptr<Matrix>> systemMatrix;

    // transfer operators that restrict only polynomial degrees
    std::vector<std::shared_ptr<PTransfer>> pTransfer;

    // transfer operators that transfer between grid levels (for piecewise linear DG!)
    std::vector<std::shared_ptr<GridTransfer>> gridTransfer;

  };

namespace Impl {
  template<class Vector, int dim>
  class TransferOperators {
    public:

      TransferOperators(MultigridData<Vector, dim>* data):
        data_(data)
      {}

      template<typename Matrix>
      void restrictMatrix(std::size_t coarseLevel, const Matrix& fine, Matrix& coarse) {
        if (coarseLevel >= data_->gridTransfer.size())
          data_->pTransfer[p_index(coarseLevel)]->galerkinRestrict(fine, coarse);
        else
          data_->gridTransfer[coarseLevel]->galerkinRestrict(fine, coarse);
      }

    private:

      std::size_t p_index(std::size_t global_idx) const {
        return global_idx - data_->gridTransfer.size();
      }

      MultigridData<Vector, dim>* data_;
  };
}
  /** Prepare multigrid data by resizing properly and
   * setting up transfer operators.
   *
   */
  template<typename Vector, typename Grid, typename M>
  auto setupData(const Grid& grid, int maximalDegree, M& matrix) {
    constexpr const int dim = Grid::dimension;
    using MGData = MultigridData<Vector, dim>;
    MGData mgData;

    const auto gridLevels = grid.maxLevel(); // number of transfers on grid +1
    const auto pLevels = static_cast<int>(std::log2(maximalDegree)); // number of transfers for p levels
    const auto levels = pLevels + gridLevels;

    mgData.systemMatrix.resize(levels+1);
    for (auto& matrix : mgData.systemMatrix)
      matrix = std::make_shared<typename MGData::Matrix>();
    mgData.systemMatrix.back() = Dune::stackobject_to_shared_ptr(matrix);

    mgData.pTransfer.resize(pLevels); // will be initialized in the next for loop

    mgData.gridTransfer.resize(gridLevels);
    for (auto& gt : mgData.gridTransfer)
      gt = std::make_shared<typename MGData::GridTransfer>();

    // setup p-transfer
    for (int l = levels - 1; l >= gridLevels; --l) {
      int currentIdx = l-gridLevels;
      int halfedOrder = maximalDegree/((pLevels-currentIdx)*2);
      auto& p = mgData.pTransfer[currentIdx];
      p=std::make_shared<typename MGData::PTransfer>();
      p->setup(*mgData.systemMatrix[l+1], halfedOrder); // setup up transfer
      p->galerkinRestrictSetOccupation(*mgData.systemMatrix[l+1], *mgData.systemMatrix[l]);
    }

    auto gTransferMats = Dune::HPDG::dgGridTransferHierarchy<M>(grid); // geometric multigrid transfer operators
    for (int l = gridLevels -1; l>=0; --l) {
      mgData.gridTransfer[l]->setup([&](auto&& matrix) {matrix=gTransferMats[l];}); // put the transfer matrix into the Transferoperator object (sigh...)
      mgData.gridTransfer[l]->galerkinRestrictSetOccupation(*mgData.systemMatrix[l+1], *mgData.systemMatrix[l]);
    }

    return mgData;
  }

  template<typename Vector, int dim>
  auto setupLevelOps(const MultigridData<Vector, dim>& data) {
    using LevelOps = typename Multigrid<Vector>::LevelOperations;
    std::vector<LevelOps> ops(data.systemMatrix.size()); // size levels+1

    ops.back().apply = operatorFromMatrix<Vector>(data.systemMatrix.back());

    // setup for p transfers
    auto maxGridLevel = data.gridTransfer.size();
    auto& matrices = data.systemMatrix;
    auto& ptransfer = data.pTransfer;
    auto& gtransfer = data.gridTransfer;

    for (int l = ops.size()-2; l >= maxGridLevel; --l) {
      auto currentIdx = l-maxGridLevel;

      ops[l].apply = operatorFromMatrix<Vector>(matrices[l]);

      ops[l+1].restrict = restrictFromMultigridTransfer<Vector>(ptransfer[currentIdx]);
      ops[l+1].prolong = prolongFromMultigridTransfer<Vector>(ptransfer[currentIdx]);
    }

    for (int l = maxGridLevel -1; l>=0; --l) {
      ops[l].apply = operatorFromMatrix<Vector>(matrices[l]);

      ops[l+1].restrict = restrictFromMultigridTransfer<Vector>(gtransfer[l]);
      ops[l+1].prolong = prolongFromMultigridTransfer<Vector>(gtransfer[l]);
    }
    using Matrix = typename MultigridData<Vector, dim>::Matrix;

    // setup smoothers
    for (std::size_t l = 0; l < ops.size(); ++l) {
      using Smoother = Dune::HPDG::DynamicBlockGS<Matrix, Vector>;
      auto smoother = std::make_shared<Smoother>(); // no memory leak because shared_ptrs will be copied
      smoother->setMatrix(*matrices[l]);
      ops[l].preSmoother = ops[l].postSmoother = smootherFromIterationStep2<Vector>(smoother);
    }

    return ops;
  }

  template<typename Vector, int dim>
  void renewMatrixHierachy(MultigridData<Vector, dim>& data) {
    auto transfer = Impl::TransferOperators<Vector, dim>(&data);

    for (int l = data.systemMatrix.size()-2; l>=0; --l) {
      transfer.restrictMatrix(l, *data.systemMatrix[l+1], *data.systemMatrix[l]);
    }
  }

  template<typename Matrix, typename Vector, int dim>
  class DGMultigridStep : public LinearIterationStep<Matrix, Vector, Solvers::DefaultBitVector_t<Vector>> {

    public:
      auto& multigridData() {
        return data_;
      }

      auto& multigrid() {
        return multigrid_;
      }

    virtual void iterate() {
      auto b = *(this->rhs_); // the MG impl. modifies the rhs, hence we make a copy here
      auto& x = *(this->x_);

      multigrid_.apply(x,b);
    }

    virtual void preprocess() {
      HPDG::MultigridSetup::renewMatrixHierachy(data_);
    }

    private:
      MultigridData<Vector, dim> data_;
      Multigrid<Vector> multigrid_;
  };

  template<typename Vector, typename Grid, typename Matrix>
  auto multigridSolver(const Grid& grid, int maximalDegree, const Matrix& matrix, int coarseIterations=5) {
    constexpr const int dim = Grid::dimension;
    auto mg = std::make_shared<DGMultigridStep<Matrix, Vector, dim>>();

    Matrix& mut_matrix = const_cast<Matrix&>(matrix);
    mg->multigridData() = setupData<Vector>(grid, maximalDegree, mut_matrix);
    mg->multigrid().levelOperations(setupLevelOps(mg->multigridData()));

    // GS coarse solver
    auto coarse_matrix_ptr = mg->multigridData().systemMatrix.front();
    auto coarse_solver = [coarse_matrix_ptr, coarseIterations](auto& x, const auto& b) {
      using Smoother = Dune::HPDG::DynamicBlockGS<Matrix, Vector>;
      auto smoother = std::make_shared<Smoother>();
      smoother->setProblem(*coarse_matrix_ptr, x, b);
      for (int i= 0; i < coarseIterations;i++)
        smoother->iterate();
    };
    mg->multigrid().coarseSolver(coarse_solver);

    return mg;
  }

}
}
}