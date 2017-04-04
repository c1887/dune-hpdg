#ifndef DUNE_HPDG_DUAL_DG_MULTIGRID_STEP
#define DUNE_HPDG_DUAL_DG_MULTIGRID_STEP

#include <dune/common/tupleutility.hh>
#include <dune/common/hybridutilities.hh>

#include <dune/istl/bcrsmatrix.hh>

#include <dune/solvers/iterationsteps/lineariterationstep.hh>
#include <dune/solvers/common/defaultbitvector.hh>
#include <dune/solvers/iterationsteps/blockgssteps.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/solvers/linearsolver.hh>
#include <dune/solvers/common/copyorreference.hh>

#include <dune/matrix-vector/axpy.hh>


#include <dune/hpdg/iterationsteps/dgmultigridstep.hh>
namespace Dune {
  namespace HPDG {
    template <class MatrixType, class VectorType, class TransferTypes, class LocalSolver, bool correctResidual = true, bool P=false, class BitVectorType = Dune::Solvers::DefaultBitVector_t<VectorType> >
    class DualDGMultigridStep : public LinearIterationStep<MatrixType, VectorType, BitVectorType>
    {
      using Base = LinearIterationStep<MatrixType, VectorType, BitVectorType>;
    public:

      //ctor
      DualDGMultigridStep(const std::array<MatrixType,2>& matrix, VectorType& x, const VectorType& rhs, const TransferTypes& transfer,
          const LocalSolver& localSolver) :
        LinearIterationStep<MatrixType, VectorType>(matrix[0], x, rhs),
        transfer_(transfer),
        penaltyMatrix_(matrix[1]),
        localGS_(localSolver){}

      /** Set factor by which the penalization is damped*/
      void setAlpha(double alpha) {
        alpha_=alpha;
      }

      /** The number of smoothing steps can be increased by a factor rate per level, i.e. coarser levels
       * get more smoothing.
       * @param rate Factor by which the smoothing steps increase when decreasing the level
       */
      void setSmoothingStepsIncreaseRate(int rate) {increaseCoarseSmoothing_ = std::max(1, rate);}
      // preprocess method (sets up coarse grid matrices and vectors)
      void preprocess() {
        // local namespace usage
        using namespace Dune::Hybrid;
        using namespace Dune::Indices;

        /* restrict the stiffness matrix to the coarser levels and resize vectors */
        forEach(integralRange(size(transfer_)),
                [&](auto reverseIndex)
        {
          const auto n = size(transfer_);
          const auto i = Dune::index_constant<n-1-reverseIndex>();
          const auto iPlusOne = Dune::index_constant<i+1>();


          // resize vectors:
          elementAt(coarseX_, i).resize(this->x_->size());
          elementAt(coarseRes_, i).resize(this->x_->size());

          // Restrict matrices
          ifElse(equals(i, Dune::index_constant<n-1>()), // on the finest level, we restrict the mat_ member, not some coarseMat_
                 [&](auto id){
            auto& transfer = elementAt(transfer_, i);
            transfer.galerkinRestrictSetOccupation(*this->mat_, id(elementAt(coarseMat_, i)));
            transfer.galerkinRestrict(*this->mat_, id(elementAt(coarseMat_, i)));
            transfer.galerkinRestrictSetOccupation(penaltyMatrix_.get(), id(elementAt(coarsePenalties_, i)));
            transfer.galerkinRestrict(penaltyMatrix_.get(), id(elementAt(coarsePenalties_, i)));
            const auto pFine = std::max(std::sqrt((double) transfer.blocksize) -1,1.0); // polynomial degree on the finer level
            const auto pCoarse = std::max(std::sqrt((double) transfer.coarseBlock) -1, 1.0); // polynomial degree on the finer level
            if (alpha_ !=0){
              elementAt(coarsePenalties_, i)*=(1.0-alpha_);
              elementAt(coarseMat_, i)-= elementAt(coarsePenalties_, i);
              elementAt(coarsePenalties_, i)*=alpha_/(1.0-alpha_);
            }
          } ,
          [&](auto id) // i != n-1
          {
            auto& transfer = elementAt(transfer_, i);
            transfer.galerkinRestrictSetOccupation(elementAt(coarseMat_, id(iPlusOne)), id(elementAt(coarseMat_, i)));
            transfer.galerkinRestrict(elementAt(coarseMat_, id(iPlusOne)), id(elementAt(coarseMat_, i)));
            transfer.galerkinRestrictSetOccupation(elementAt(coarsePenalties_, id(iPlusOne)), id(elementAt(coarsePenalties_, i)));
            transfer.galerkinRestrict(elementAt(coarsePenalties_, id(iPlusOne)), id(elementAt(coarsePenalties_, i)));
            const auto pFine = std::max(std::sqrt((double) transfer.blocksize) -1, 1.0); // polynomial degree on the finer level
            const auto pCoarse = std::max(std::sqrt((double)transfer.coarseBlock) -1, 1.0); // polynomial degree on the finer level
            if (alpha_ !=0){
              elementAt(coarsePenalties_, i)*=(1.0-alpha_);
              elementAt(coarseMat_, i)-= elementAt(coarsePenalties_, i);
              elementAt(coarsePenalties_, i)*=alpha_/(1.0-alpha_);
            }
          });
        });
      }

      void setProblem(std::array<const MatrixType, 2ul>& m, VectorType& x, const VectorType& rhs) {
        Base::setProblem(m[0], x, rhs);
        penaltyMatrix_.get()=m[1];
      }
      // TODO: Smart ptr!
      template<class S>
      void setBaseSolver(S& s) {basesolver_ = &s;}

      void iterate() {
        for (int i=0; i < coarseCorrections_; i++)
          vcycle(preSmoothingSteps_, postSmoothingsSteps_);
      }

      void setMGType(int coarseCorrections, int preSmoothingSteps, int postSmoothingsSteps) {
        coarseCorrections_ = coarseCorrections;
        preSmoothingSteps_ = preSmoothingSteps;
        postSmoothingsSteps_ = postSmoothingsSteps;
      }

      template<class Index>
      auto returnMatrix (const Index i) {
        return Dune::Hybrid::elementAt(coarseMat_, i);
      }

      template<class V, class Index>
      VectorType prolongVector(const V& v, const Index i) {
        using namespace Dune::Hybrid;

        CoarseVectors hierarchy;
        elementAt(hierarchy, i) = v;
        const auto n = size(transfer_);
        const auto lastIndex = Dune::index_constant<n-1>();
        VectorType ret;
        ret.resize(elementAt(transfer_,lastIndex).getMatrix().N());
        ret = 0;

        forEach(integralRange(i, n), [&](auto j)
        {
          ifElse(equals(j,lastIndex),[&](auto id) {
              const auto jj = id(j);
              const auto& last = id(elementAt(hierarchy, jj));
              auto& next = id(ret);
              id(elementAt(transfer_, jj)).prolong(last, next);
            }
            ,[&] (auto id) {
              const auto jj = id(j);
              const auto jPlusOne= Dune::index_constant<jj+1>();
              const auto& last= id(elementAt(hierarchy, jj));
              auto& next = id(elementAt(hierarchy, jPlusOne));
              next.resize(elementAt(transfer_,lastIndex).getMatrix().N());
              id(elementAt(transfer_, jj)).prolong(last, next);
          });
        });
return ret;
      }

    private:

      void vcycle(int preSmoothing, int postSmoothing) {


        using namespace Dune::Hybrid; // use namespace locally
        using namespace Dune::Indices; // use namespace locally

        const auto n = size(transfer_);
        const auto nMinusOne = Dune::index_constant<n-1>();


        auto globalSmoother = BlockGaussSeidel<MatrixType, VectorType, BitVectorType>::create(localGS_);
        globalSmoother.setProblem(*this->mat_, *this->x_, *this->rhs_);

        // Presmoothing
        for (int iter=0; iter<preSmoothing; iter++) {
          globalSmoother.iterate();
          //globalSmoother.toggleDirection(); // This is non standard dune-solvers and hence deactivated here
        }

        // restrict residual to first coarse level
        { // block scope for not carrying around restrictedCoarseResidual all the time
          auto restrictedCoarseResidual = *this->rhs_;
          Dune::MatrixVector::subtractProduct(restrictedCoarseResidual, *this->mat_, *this->x_);

          const auto& fineTransfer = elementAt(transfer_, nMinusOne);
          auto& fineRes = elementAt(coarseRes_, nMinusOne);
          fineTransfer.restrict(restrictedCoarseResidual, fineRes);
        }

        // First (downward) part of the V-cycle
        forEach(integralRange(size(transfer_)), [&, n=n](auto reverseIndex)
        {
          preSmoothing*=increaseCoarseSmoothing_;
          const auto i = Dune::index_constant<n-1-reverseIndex>();

          // define some aliases for convenience
          const auto& matrix = elementAt(coarseMat_, i);
          auto& res = elementAt(coarseRes_, i);
          auto& x = elementAt(coarseX_, i);


          using Matrix = typename std::tuple_element<i, CoarseMatrices>::type;
          using Vector = typename std::tuple_element<i, CoarseVectors>::type;
          using BitVec = Dune::Solvers::DefaultBitVector_t<Vector>;

          auto newRes = res; // residual that will be updated after smoothing and restricted to the coarser level

          /* set up smoother */
          x = 0; // no old corrections
          x.resize(matrix.N());
          using Smoother = BlockGaussSeidel<Matrix, Vector, BitVec>; // GS outer solver
          auto smoother = Smoother::create(localGS_);

          /* smooth preSmoothing times */
          smoother.setProblem(matrix, x, res);
          if (i != 0 or basesolver_==nullptr) {
            for (int iter=0; iter< preSmoothing; ++iter) {
              smoother.iterate();
              //smoother.toggleDirection(); // This is non standard dune-solvers and hence deactivated here
            }

            /* update residual */
            if (correctResidual) Dune::MatrixVector::subtractProduct(newRes, matrix, x);
          }

          /* restrict residual to coarser level*/
          ifElse(equals(i, _0), [&](auto id)
          {
            if (basesolver_!=nullptr) {
              auto ii = id(i);
              using Matrix = typename std::tuple_element<ii, CoarseMatrices>::type;
              using Vector = typename std::tuple_element<ii, CoarseVectors>::type;

              x.resize(matrix.M());

              typedef ::LoopSolver<Vector> DuneSolversLoopSolver;

              if (dynamic_cast<DuneSolversLoopSolver*>(this->basesolver_)) {

                DuneSolversLoopSolver* loopBaseSolver = dynamic_cast<DuneSolversLoopSolver*> (this->basesolver_);

                typedef LinearIterationStep<Matrix, Vector> SmootherType;
                assert(dynamic_cast<SmootherType*>(loopBaseSolver->iterationStep_));

                dynamic_cast<SmootherType*>(loopBaseSolver->iterationStep_)->setProblem(matrix, x, res);
                dynamic_cast<SmootherType*>(loopBaseSolver->iterationStep_)->preprocess();
              }
              else if (dynamic_cast<LinearSolver<Matrix, Vector>*>(this->basesolver_)) {

                LinearSolver<Matrix, Vector>* linearBaseSolver = dynamic_cast<LinearSolver<Matrix, Vector>*> (this->basesolver_);

                linearBaseSolver->setProblem(matrix, x, res);
              }

              else {
                DUNE_THROW(SolverError, "You can't use " << typeid(*this->basesolver_).name()
                    << " as a base solver in a MultigridStep!");
              }

              basesolver_->solve();
            }
          },
          //else
          [&](auto id)
          {
            auto ii =id(i);
            const auto iMinusOne= Dune::index_constant<ii-1>();
            id(elementAt(id(transfer_), id(iMinusOne))).restrict(newRes, elementAt(coarseRes_, id(iMinusOne)));
          });
        });

        postSmoothing*=std::pow(increaseCoarseSmoothing_, n.value);
        // Second (upward) part of the V-cycle
        forEach(integralRange(n), [&](auto i)
        {
          // define some aliases for convenience
          const auto& matrix = elementAt(coarseMat_, i);
          auto& res = elementAt(coarseRes_, i);
          auto& x = elementAt(coarseX_, i);

          using Matrix = typename std::tuple_element<i, CoarseMatrices>::type;
          using Vector = typename std::tuple_element<i, CoarseVectors>::type;
          using BitVec = Dune::Solvers::DefaultBitVector_t<Vector>;

          //const int p = (int) std::sqrt((int) Matrix::block_type::rows) -1;
          /* set up smoother */
          if (i!=0 or basesolver_==nullptr) {
            using Smoother = BlockGaussSeidel<Matrix, Vector, BitVec>; // GS outer solver
            auto smoother = Smoother::create(localGS_, Dune::Solvers::BlockGS::Direction::BACKWARD);
            smoother.setProblem(matrix, x, res);

            /* smooth postSmoothing times */
            for (int iter=0; iter< postSmoothing; ++iter) {
              smoother.iterate();
              //smoother.toggleDirection(); // This is non standard dune-solvers and hence deactivated here
            }
          }

          // prolong corrections and add them to the top level
          ifElse(equals(i, nMinusOne), [&](auto id) {     // if (i==n-1)
            VectorType fineGridCorrection(this->x_->size()); // global VectorType
            fineGridCorrection=0;

            // prolong x to the higher level
            id(elementAt(transfer_, id(i))).prolong(x, fineGridCorrection);
            // add to the global iterate *x_
            *this->x_+=fineGridCorrection;
          },
          //else
          [&](auto id)
          {
            auto ii = id(i); // make ugly copy for technical reasons
            const auto iPlusOne = Dune::index_constant<ii+1>();
            auto& fineX = id(elementAt(coarseX_, id(iPlusOne)));

            auto fineGridCorrection = fineX;
            fineGridCorrection = 0;
            id(elementAt(transfer_, ii)).prolong(x, fineGridCorrection);
            fineX += fineGridCorrection;
          }
          );
          postSmoothing/=increaseCoarseSmoothing_;
        });

        //post smoothing
        for (int iter=0; iter<postSmoothing; iter++) {
          globalSmoother.iterate();
          //globalSmoother.toggleDirection(); // This is non standard dune-solvers and hence deactivated here
        }
      }

      using CoarseMatrices = typename Dune::ForEachType<CoarseMatrixTypeEvaluator, TransferTypes>::Type;
      using CoarseVectors = typename Dune::ForEachType<VectorTypeEvaluator, TransferTypes>::Type;

      template<class M, class V, class B>
      using BlockGaussSeidel = typename Dune::Solvers::BlockGSStepFactory<M, V, B>;

      // the fine matrix, rhs and iterate are taken care of by the base class
      const TransferTypes& transfer_;
      CoarseMatrices coarseMat_;
      CoarseMatrices coarsePenalties_;
      CoarseVectors coarseX_;
      CoarseVectors coarseRes_;
      LocalSolver localGS_;
      Solver* basesolver_ = nullptr;
      Dune::Solvers::CopyOrReference<MatrixType> penaltyMatrix_;

      int coarseCorrections_ = 1; // number of MG cycles
      int preSmoothingSteps_ = 3;
      int postSmoothingsSteps_ = 3;

      double alpha_=0;
      int increaseCoarseSmoothing_ = 1;

    };
  } // namespace HPDG 
} // namespace Dune
#endif //DUNE_HPDG_DUAL_DG_MULTIGRID_STEP
