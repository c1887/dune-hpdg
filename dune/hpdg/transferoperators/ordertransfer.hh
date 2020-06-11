#ifndef DUNE_HPDG_DG_MULTIGRID_ORDER_TRANSFER_HH
#define DUNE_HPDG_DG_MULTIGRID_ORDER_TRANSFER_HH

#include <memory>
#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/istl/bdmatrix.hh>

#include <dune/matrix-vector/transformmatrix.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/transferoperators/dynamicordertransfer.hh>
#include <dune/hpdg/transferoperators/arithmetic.hh>


namespace Dune {
  namespace HPDG {

/** \brief Galerkin restriction and prolongation for Discontinuous Galerkin stiffness matrices
 *
 * This class implements the Galerkin restriciton and prolongation between DG bases of different orders on the same grid.
 * Matrices and vectors have to be blocked element-wise(!). This leads to very efficient computations since the transfer matrices are
 * block diagonal.
 */
    template<class VectorType, int dim>
    class DGOrderTransfer
    {


      using DynamicMatrixType = DynamicBCRSMatrix<FieldMatrix<typename VectorType::field_type,1,1>>;
      using MatrixType = DynamicMatrixType;

      using BlockType = typename MatrixType::block_type;
    public:

      using field_type = typename VectorType::field_type;

      using TransferOperatorType = DynamicMatrixType;

      virtual ~DGOrderTransfer() {}

      template<typename M>
      void setup(const M& fineReference, int maxOrder)
      {
        // set up index set (block diagonal):
        {
          MatrixIndexSet idx(fineReference.N(), fineReference.N());
          for (size_t i = 0; i < fineReference.N(); i++)
            idx.add(i,i);
          idx.exportIdx(matrix_);
          matrix_.finishIdx();
        }

        // set block dimensions
        auto maxBlockSize = Impl::orderToBlockSize(maxOrder, dim);
        for (size_t i = 0; i < fineReference.N(); i++) {
          const auto& blocks = fineReference.blockRows(i);
          matrix_.blockRows(i)=blocks;
          matrix_.blockColumns(i)=std::min(blocks, maxBlockSize);
        }
        // allocate memory
        matrix_.update();

        // fill blocks
        using LocalMatrix = Dune::DynamicMatrix<field_type>;
        auto transferCache = Impl::TransferMatrixCache<LocalMatrix, dim>{};
        for (size_t i = 0; i < fineReference.N(); i++) {
          auto& transferBlock = matrix_[i][i];
          auto blocks = fineReference.blockRows(i);
          // case 1: block is not too large, we can put an identity
          if (blocks<=maxBlockSize) {
            for (size_t i = 0; i < transferBlock.N(); i++)
              for (size_t j = 0; j < transferBlock.M(); j++)
                transferBlock[i][j]= (i==j) ? 1.0 : 0.0;
          }
          // case 2: block (and hence order) too large, put transferblock to maxOrder here
          else {
            const auto& to = transferCache(maxOrder, Impl::blockSizeToOrder(blocks, dim));
            DUNE_ASSERT_BOUNDS(to.N() == transferBlock.N());
            DUNE_ASSERT_BOUNDS(to.M() == transferBlock.M());

            transferBlock = to;
          }
        }
      }

      /** \brief Restrict a function from the fine onto the coarse grid
     */
      void restrict(const VectorType& fineVector, VectorType& coarseVector) const
      {
        coarseVector.setSize(matrix_.M());
        // step 1: prepare coarse vector:
        for (size_t i = 0; i < matrix_.M(); i++)
          coarseVector.blockRows(i)=matrix_.blockColumns(i);
        coarseVector.update();

        // step 2: restrict
        coarseVector = 0.;
        Arithmetic::transposedMatrixVectorProduct(matrix_, fineVector, coarseVector);
      }

      /** \brief Prolong a function from the coarse onto the fine grid
      */
      void prolong(const VectorType& coarseVector, VectorType& fineVector) const
      {
        // fine vector should have been used already, but we do not risk anything and
        // prepare again:
        fineVector.setSize(coarseVector.size());
        for (size_t i = 0; i < fineVector.size(); i++)
          fineVector.blockRows(i)=matrix_.blockRows(i);
        fineVector.update();


        // prolong
        fineVector = 0.;
        Arithmetic::matrixVectorProduct(matrix_, coarseVector, fineVector);
      }

      /** \brief Galerkin assemble a coarse stiffness matrix
      */
      template<typename M>
      void galerkinRestrict(const M& fineMatrix, M& coarseMatrix) const
      {

        const auto& fineMat = fineMatrix;
        auto& coarseMat = coarseMatrix;
        coarseMat = 0;

        // code from block transfer. TODO Check it carefully!
        for (std::size_t i =0; i< fineMat.N(); i++) {
          const auto& Ai = fineMat[i];
          Dune::MatrixVector::sparseRangeFor(Ai, [&](auto&& Aij, auto&& j) {
            const auto& Ti = matrix_[i];
            const auto& Tj = matrix_[j];
            Dune::MatrixVector::sparseRangeFor(Ti, [&](auto&& Tik, auto&& k) {
              Dune::MatrixVector::sparseRangeFor(Tj, [&](auto&& Tjl, auto&& l) {
                  Dune::HPDG::Arithmetic::addTransformedMatrix(coarseMat[k][l], Tik, Aij, Tjl);
              });
            });
          });
        }
      }

      /** \brief Set Occupation of Galerkin restricted coarse stiffness matrix
    *
    * Set occupation of Galerkin restricted coarse matrix. Call this one before
    * galerkinRestrict to ensure all non-zeroes are present
    * \param fineMat The fine level matrix
    * \param coarseMat The coarse level matrix
    */
      template<typename M>
      void galerkinRestrictSetOccupation(const M& fineMat, M& coarseMat) const
      {
        // This one is easy, as the transfer operator has block diag. structure. Hence,
        // the coarse matrix will have the same block structure as the fine matrix.

        Dune::MatrixIndexSet idx(fineMat.N(), fineMat.M());
        idx.import(fineMat);
        idx.exportIdx(coarseMat);

        coarseMat.finishIdx();

        for (size_t i = 0; i < fineMat.N(); i++) {
          coarseMat.blockRows(i)=coarseMat.blockColumns(i)=matrix_.blockColumns(i);
        }
        coarseMat.update();

      }

      /** \brief Direct access to the operator matrix, if you absolutely want it! */
      const TransferOperatorType& getMatrix() const {
        return matrix_;
      }

    protected:

      TransferOperatorType matrix_;

    };
  } // end namespace HPDG
} // end namespace Dune

#endif
