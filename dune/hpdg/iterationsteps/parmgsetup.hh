#pragma once

#include <memory>

#include <dune/parmg/iterationstep/multigrid.hh>
#include <dune/hpdg/iterationsteps/dynamicblockgs.hh>
#include <dune/hpdg/iterationsteps/l1smoother.hh>
#include <dune/hpdg/iterationsteps/coarsegrid.hh>
#include <dune/hpdg/parallel/communicationhpdg.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>

#include <dune/hpdg/transferoperators/ordertransfer.hh>
#include <dune/hpdg/transferoperators/dynamicordertransfer.hh>
#include <dune/hpdg/transferoperators/dynamicblocktransfer.hh>

#include <dune/hpdg/assemblers/dgtodggridtransferassembler.hh>
#include <dune/solvers/common/defaultbitvector.hh>

#include <dune/common/power.hh>
#include <dune/common/unused.hh>

#include <dune/istl/matrixindexset.hh>


namespace Dune {
namespace HPDG {
namespace MultigridSetup {
  //using namespace ParMG; // ouchie
  template<typename Vector, int dim>
  struct MultigridData {

    using Matrix = DynamicBCRSMatrix<FieldMatrix<double,1,1>>;
    using PTransfer = Dune::HPDG::DGOrderTransfer<Vector, dim>;
    using GridTransfer = DynamicBlockTransferOperator<Vector>;

    int ranks;

    // system matrices on each level
    std::vector<std::shared_ptr<Matrix>> systemMatrix;

    // transfer operators that restrict only polynomial degrees
    std::vector<std::shared_ptr<PTransfer>> pTransfer;

    // transfer operators that transfer between grid levels (for piecewise linear DG!)
    std::vector<std::shared_ptr<GridTransfer>> gridTransfer;

    std::vector<std::shared_ptr<ParMG::CommHPDG>> comms;

    std::vector<std::vector<std::size_t>> ghostsPerLevel;

    /* if needed, this can hold the _global_ coarse Matrix
     * restricted to rank 0.
     * This might be useful for a proper coarse grid
     * correction.
     */
    std::shared_ptr<Matrix> rank0CoarseMatrix;


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
    mgData.ranks = grid.comm().size();

    const auto gridLevels = grid.maxLevel(); // number of transfers on grid
    const auto pLevels = static_cast<int>(std::log2(maximalDegree)); // number of transfers for p levels
    const auto levels = pLevels + gridLevels; // actually, number of transfers

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

    // setup ghost indicators
    mgData.ghostsPerLevel.resize(levels+1);
    using Basis = ::Impl::MultilevelBasis<Grid>;
    Basis basis(grid);
    for(std::size_t i = 0; i < grid.maxLevel(); i++) {
      mgData.ghostsPerLevel[i] = basis.nonInteriorElements(i);
    }

    for(std::size_t i = grid.maxLevel(); i < levels+1; i++) {
      mgData.ghostsPerLevel[i] = basis.nonInteriorElements(grid.maxLevel());
    }

    return mgData;
  }


  template<typename Vector, int dim>
  auto setupLevelOps(const MultigridData<Vector, dim>& data) {
    using LevelOps = typename ParMG::Multigrid<Vector>::LevelOperations;
    std::vector<LevelOps> ops(data.systemMatrix.size()); // size levels+1

    ops.back().apply = ParMG::operatorFromMatrix<Vector>(data.systemMatrix.back());

    // setup for p transfers
    auto maxGridLevel = data.gridTransfer.size();
    auto& matrices = data.systemMatrix;
    auto& ptransfer = data.pTransfer;
    auto& gtransfer = data.gridTransfer;

    for (int l = ops.size()-2; l >= maxGridLevel; --l) {
      auto currentIdx = l-maxGridLevel;

      ops[l].apply = ParMG::operatorFromMatrix<Vector>(matrices[l]);

      ops[l+1].restrict = ParMG::restrictFromMultigridTransfer<Vector>(ptransfer[currentIdx]);
      ops[l+1].prolong = ParMG::prolongFromMultigridTransfer<Vector>(ptransfer[currentIdx]);
    }

    for (int l = maxGridLevel -1; l>=0; --l) {
      ops[l].apply = ParMG::operatorFromMatrix<Vector>(matrices[l]);

      ops[l+1].restrict = ParMG::restrictFromMultigridTransfer<Vector>(gtransfer[l]);
      ops[l+1].prolong = ParMG::prolongFromMultigridTransfer<Vector>(gtransfer[l]);
    }
    using Matrix = typename MultigridData<Vector, dim>::Matrix;

    // setup smoothers
    for (std::size_t l = 0; l < ops.size(); ++l) {
      //using Smoother = Dune::HPDG::DynamicBlockGS<Matrix, Vector>;
      using Smoother = Dune::HPDG::L1Smoother<Matrix, Vector>;
      auto smoother = std::make_shared<Smoother>(data.ghostsPerLevel[l]); // no memory leak because shared_ptrs will be copied
      //auto smoother = std::make_shared<Smoother>(); // no memory leak because shared_ptrs will be copied
      smoother->setMatrix(*matrices[l]);
      smoother->preprocess();
      ops[l].preSmoother = ops[l].postSmoother = ParMG::smootherFromSolversIterationStep<Vector>(smoother);
      //ops[l].preSmoother = ops[l].postSmoother = ParMG::smootherFromSolversIterationStep<Vector>(smoother, 1./data.ranks);
    }

    return ops;
  }

  template<typename Grid>
  auto setupComms(const Grid& grid) {
    std::vector<std::shared_ptr<ParMG::CommHPDG>> comms(grid.maxLevel()+1);
    using Basis = ::Impl::MultilevelBasis<Grid>;
    Basis basis(grid);
    for (int i = 0; i <= grid.maxLevel(); ++i) {
      using T = FieldVector<double, 1>;
      comms[i] = ParMG::makeDGInterface<T>(basis, i);
    }

    return comms;
  }

// TODO Remove?
  template<typename Grid>
  auto setupLeafComm(const Grid& grid) {
    std::shared_ptr<ParMG::CommHPDG> comm;
    auto basis = ParMG::Impl::HPDG::LeafBasis<Grid>(grid);
    using T = FieldVector<double, 1>;
    comm = ParMG::makeDGInterface<T>(basis, grid.maxLevel());
    return comm;
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
      // update fine matrix
      data_.systemMatrix.back() = std::const_pointer_cast<Matrix>(this->mat_);

      /* if fine matrix was changed, the `apply` part of the level operations
       * on the finest level needs to be updated. Unfortunately, the level ops are not exposed,
       * therefore we have to reset the all of them via this setup tool:
       */

      auto ops= setupLevelOps(data_);

      auto n_levels = data_.gridTransfer.size()+1;
      for(std::size_t i = 0; i < n_levels; i++) {
        ops[i].restrictToMaster = ParMG::makeDGRestrict<Vector>(*(data_.comms[i]));
        ops[i].accumulate = ParMG::makeDGAccumulate<Vector>(*(data_.comms[i]));
        ops[i].collect = ParMG::makeDGCollect<Vector>(*(data_.comms[i]));
        ops[i].copyFromMaster = ParMG::makeDGCopy<Vector>(*(data_.comms[i]));
      }

      for(std::size_t i = n_levels; i< ops.size(); ++i) {
        ops[i].restrictToMaster = ParMG::makeDGRestrict<Vector>(*(data_.comms.back()));
        ops[i].accumulate = ParMG::makeDGAccumulate<Vector>(*(data_.comms.back()));
        ops[i].collect = ParMG::makeDGCollect<Vector>(*(data_.comms.back()));
        ops[i].copyFromMaster = ParMG::makeDGCopy<Vector>(*(data_.comms.back()));
      }
      multigrid_.levelOperations(ops);

      // recalculate matrix hierarchy
      HPDG::MultigridSetup::renewMatrixHierachy(data_);
    }

    private:
      MultigridData<Vector, dim> data_;
      ParMG::Multigrid<Vector> multigrid_;
  };

  template<typename Matrix, typename Vector, int dim>
  auto gaussSeidelCoarseSolver(std::shared_ptr<DGMultigridStep<Matrix, Vector, dim>> mg, int coarseIterations = 5) {
    /// GS coarse solver

    /* If the shared_ptr "mg" is copied here, we get a circular references.
     * Therefore, use a weak ptr
     */
    auto weak_mg = std::weak_ptr<typename decltype(mg)::element_type>(mg);
    auto coarse_solver = [weak_mg, coarseIterations](Vector& x, const Vector& b, bool, const Vector&) {
      if (weak_mg.expired())
        return;
      auto local_mg = weak_mg.lock(); // get shared_ptr from weak_ptr
      const auto& op = local_mg->multigrid().levelOperations().front();
      auto coarse_matrix_ptr = local_mg->multigridData().systemMatrix.front();
      using Smoother = Dune::HPDG::DynamicBlockGS<Matrix, Vector>;
      auto smoother = Smoother();

      auto x_iter = x;
      x_iter =0.;

      auto r = b;

      smoother.setProblem(*coarse_matrix_ptr, x_iter, r);

      auto tmp = x;

      for (int i= 0; i < coarseIterations;i++) {
        smoother.iterate();
        op.maybeCopyFromMaster(x_iter);
        // damping:
        x_iter *= 0.8;
        x+=x_iter;
        coarse_matrix_ptr->mv(x_iter, tmp);
        op.maybeRestrictToMaster(tmp);
        r-=tmp;
      }
    };
    return coarse_solver;
  }

  template<typename Matrix, typename Vector, int dim>
  auto l1CoarseSolver(std::shared_ptr<DGMultigridStep<Matrix, Vector, dim>> mg, int coarseIterations = 5) {
    /// GS coarse solver

    /* If the shared_ptr "mg" is copied here, we get a circular references.
     * Therefore, use a weak ptr
     */
    auto weak_mg = std::weak_ptr<typename decltype(mg)::element_type>(mg);
    auto coarse_solver = [weak_mg, coarseIterations](Vector& x, const Vector& b, bool, const Vector&) {
      if (weak_mg.expired())
        return;
      auto local_mg = weak_mg.lock(); // get shared_ptr from weak_ptr
      const auto& op = local_mg->multigrid().levelOperations().front();
      auto coarse_matrix_ptr = local_mg->multigridData().systemMatrix.front();
      using Smoother = Dune::HPDG::L1Smoother<Matrix, Vector>;
      auto smoother = Smoother(local_mg->multigridData().ghostsPerLevel.front());

      auto x_iter = x;
      x_iter =0.;

      auto r = b;

      smoother.setProblem(*coarse_matrix_ptr, x_iter, r);
      smoother.preprocess();

      auto tmp = x;

      for (int i= 0; i < coarseIterations;i++) {
        smoother.iterate();
        op.maybeCopyFromMaster(x_iter);
        // damping:
        //x_iter *= 0.8;
        x+=x_iter;
        coarse_matrix_ptr->mv(x_iter, tmp);
        op.maybeRestrictToMaster(tmp);
        r-=tmp;
      }
    };
    return coarse_solver;
  }

  /** \brief This plugs together the coarse matrices to a single
   * _global_ coarse matrix and sets the correspondin object
   * only on process 0;
   * */
  template<typename Grid, typename Matrix, typename Vector, int dim>
  void globalCoarseMatrix(const Grid& grid, DGMultigridStep<Matrix, Vector, dim>& mg) {
    auto matrix = Matrix();
    auto rank = grid.comm().rank();
    auto basis = ParMG::Impl::HPDG::LevelBasis<Grid>(grid, 0);
    auto dofs = ParMG::Impl::makeGlobalDofHPDG(basis, 0);
    const auto& global = dofs.first;
    const auto& owner = dofs.second;

    auto redist = Rank0Collector(grid.comm(), global, owner);

    // setup index set
    auto matrixIdx = Dune::MatrixIndexSet();

    redist.restrictMatrix(*mg.multigridData().systemMatrix.front(), matrix);

    if (rank==0)
      mg.multigridData().rank0CoarseMatrix = std::make_shared<Matrix>(std::move(matrix));
  }

  template<typename Grid, typename Vector>
  auto globalCoarseVector(const Grid& grid, const Vector& vector) {
    auto result = Vector();

    auto basis = ParMG::Impl::HPDG::LevelBasis<Grid>(grid, 0);
    auto dofs = ParMG::Impl::makeGlobalDofHPDG(basis, 0);
    const auto& global = dofs.first;
    const auto& owner = dofs.second;

    auto redist = Rank0Collector(grid.comm(), global, owner);
    redist.restrictVector(vector, result);

    return result;
  }


  template<typename Grid, typename Matrix, typename Vector, int dim>
  auto singleProcessGaussSeidel(const Grid& grid, std::shared_ptr<DGMultigridStep<Matrix, Vector, dim>> mg, int coarseIterations = 5) {
    int rank = grid.comm().rank();
    /// GS coarse solver where the matrix is collected on process 0.

    /* If the shared_ptr "mg" is copied here, we get a circular references.
     * Therefore, use a weak ptr
     */
    auto weak_mg = std::weak_ptr<typename decltype(mg)::element_type>(mg);
    auto coarse_solver = [&grid, weak_mg, rank, coarseIterations](Vector& x, const Vector& b, bool matrixChanged, const Vector&) {
      if (weak_mg.expired())
        return;
      auto local_mg = weak_mg.lock(); // get shared_ptr from weak_ptr
      auto coarse_matrix_ptr = local_mg->multigridData().systemMatrix.front();
      using Smoother = Dune::HPDG::DynamicBlockGS<Matrix, Vector>;
      //auto smoother = std::make_shared<Smoother>();
      auto smoother = Smoother();
      if (matrixChanged) { // TODO: das wird scheinbar in jeder neuen Iteration auf true gesetzt. Das will man auch bei TNNMG
        globalCoarseMatrix(grid, *local_mg);
      }

      auto global_x = globalCoarseVector(grid, x);
      auto global_b = globalCoarseVector(grid, b);

      if (rank==0) {
        auto x_iter = global_x;
        x_iter =0.;

        auto r = global_b;

        smoother.setProblem(*local_mg->multigridData().rank0CoarseMatrix, x_iter, r);

        auto tmp = x;

        for (int i= 0; i < coarseIterations;i++) {
          smoother.iterate();
          //local_mg->multigridData().rank0CoarseMatrix->mmv(x_iter, r);
          //x_iter=0.;
        }
          global_x+=x_iter;
      }
      grid.comm().barrier();

      auto basis = ParMG::Impl::HPDG::LevelBasis<Grid>(grid, 0);
      auto dofs = ParMG::Impl::makeGlobalDofHPDG(basis, 0);
      const auto& global = dofs.first;
      const auto& owner = dofs.second;

      auto redist = Rank0Collector(grid.comm(), global, owner);
      redist.scatterVector(global_x, x);
    };
    return coarse_solver;
  }

  template<typename Vector, typename Grid, typename Matrix>
  auto multigridSolver(const Grid& grid, int maximalDegree, const Matrix& matrix) {
    constexpr const int dim = Grid::dimension;
    auto mg = std::make_shared<DGMultigridStep<Matrix, Vector, dim>>();

    Matrix& mut_matrix = const_cast<Matrix&>(matrix);
    mg->multigridData() = setupData<Vector>(grid, maximalDegree, mut_matrix);
    mg->multigridData().comms = setupComms(grid);

    auto ops = setupLevelOps(mg->multigridData());

    // set comms in the level transfers:
    for(std::size_t i = 0; i < grid.maxLevel(); i++) {
      ops[i].restrictToMaster = ParMG::makeDGRestrict<Vector>(*(mg->multigridData().comms[i]));
      ops[i].accumulate = ParMG::makeDGAccumulate<Vector>(*(mg->multigridData().comms[i]));
      ops[i].collect = ParMG::makeDGCollect<Vector>(*(mg->multigridData().comms[i]));
      ops[i].copyFromMaster = ParMG::makeDGCopy<Vector>(*(mg->multigridData().comms[i]));
    }
    // and for the rest
    for(std::size_t i = grid.maxLevel(); i< ops.size(); ++i) {
      ops[i].restrictToMaster = ParMG::makeDGRestrict<Vector>(*(mg->multigridData().comms.back()));
      ops[i].accumulate = ParMG::makeDGAccumulate<Vector>(*(mg->multigridData().comms.back()));
      ops[i].collect = ParMG::makeDGCollect<Vector>(*(mg->multigridData().comms.back()));
      ops[i].copyFromMaster = ParMG::makeDGCopy<Vector>(*(mg->multigridData().comms.back()));
    }

    mg->multigrid().levelOperations(ops);

    return mg;
  }

}
}
}
