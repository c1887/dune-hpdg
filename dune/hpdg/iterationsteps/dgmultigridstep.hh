#ifndef DUNE_SOLVERS_DG_MULTIGRID_STEP
#define DUNE_SOLVERS_DG_MULTIGRID_STEP

#include <dune/common/tupleutility.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/typetraits.hh>

#include <dune/istl/bcrsmatrix.hh>

#include <dune/solvers/iterationsteps/lineariterationstep.hh>
#include <dune/solvers/common/defaultbitvector.hh>
#include <dune/solvers/iterationsteps/blockgssteps.hh>
#include <dune/solvers/solvers/loopsolver.hh>
#include <dune/solvers/solvers/linearsolver.hh>

#include <dune/matrix-vector/axpy.hh>


namespace { // anonymous namespace for implementation details

  // Helper template for 'ForEachType'
  template<class T>
  struct CoarseMatrixTypeEvaluator {
    typedef Dune::BCRSMatrix<Dune::FieldMatrix<double, T::coarseBlock, T::coarseBlock> > Type; // \todo change BCRS for sth. more general
  };

  // Helper template for 'ForEachType'
  template<class T>
  struct VectorTypeEvaluator {
    typedef Dune::BlockVector<Dune::FieldVector<double, T::coarseBlock> > Type; // \todo change BlockVector for sth. more general
  };

} // anonymous namespace

namespace Dune {
  namespace Solvers {
    template <class MatrixType, class VectorType, class TransferTypes, class LocalSolver, class BitVectorType = Dune::Solvers::DefaultBitVector_t<VectorType> >
    class DGMultigridStep : public LinearIterationStep<MatrixType, VectorType, BitVectorType>
    {
    public:

      //ctor
      DGMultigridStep(const MatrixType& matrix, VectorType& x, const VectorType& rhs, const TransferTypes& transfer, const LocalSolver& localSolver) :
        LinearIterationStep<MatrixType, VectorType>(matrix, x, rhs),
        transfer_(transfer),
        localGS_(localSolver){}

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
            elementAt(transfer_, i).galerkinRestrictSetOccupation(*this->mat_, id(elementAt(coarseMat_, i)));
            elementAt(transfer_, i).galerkinRestrict(*this->mat_, id(elementAt(coarseMat_, i)));
          } ,
          [&](auto id) // i != n-1
          {
            elementAt(transfer_, i).galerkinRestrictSetOccupation(elementAt(coarseMat_, id(iPlusOne)), elementAt(coarseMat_, i));
            elementAt(transfer_, i).galerkinRestrict(elementAt(coarseMat_, id(iPlusOne)), elementAt(coarseMat_, i));
          });
        });
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
            for (int iter=0; iter< preSmoothing; ++iter)
              smoother.iterate();

            /* update residual */
            Dune::MatrixVector::subtractProduct(newRes, matrix, x);
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
                assert(dynamic_cast<SmootherType*>(&(loopBaseSolver->getIterationStep())));

                dynamic_cast<SmootherType*>(&(loopBaseSolver->getIterationStep()))->setProblem(matrix, x, res);
                dynamic_cast<SmootherType*>(&(loopBaseSolver->getIterationStep()))->preprocess();
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

          /* set up smoother */
          if (i!=0 or basesolver_==nullptr) {
            using Smoother = BlockGaussSeidel<Matrix, Vector, BitVec>; // GS outer solver
            auto smoother = Smoother::create(localGS_);
            smoother.setProblem(matrix, x, res);

            /* smooth postSmoothing times */
            for (int iter=0; iter< postSmoothing; ++iter)
              smoother.iterate();
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
        });

        //post smoothing
        for (int iter=0; iter<postSmoothing; iter++)
          globalSmoother.iterate();
      }

      using CoarseMatrices = typename Dune::ForEachType<CoarseMatrixTypeEvaluator, TransferTypes>::Type;
      using CoarseVectors = typename Dune::ForEachType<VectorTypeEvaluator, TransferTypes>::Type;

      template<class M, class V, class B>
      using BlockGaussSeidel = typename Dune::Solvers::BlockGSStepFactory<M, V, B>;


      // the fine matrix, rhs and iterate are taken care of by the base class
      const TransferTypes& transfer_;
      CoarseMatrices coarseMat_;
      CoarseVectors coarseX_;
      CoarseVectors coarseRes_;
      LocalSolver localGS_;
      Solver* basesolver_ = nullptr;

      int coarseCorrections_ = 1; // number of MG cycles
      int preSmoothingSteps_ = 3;
      int postSmoothingsSteps_ = 3;

    };
  } // namespace Solvers
} // namespace Dune
#endif //DUNE_SOLVERS_DG_MULTIGRID_STEP
