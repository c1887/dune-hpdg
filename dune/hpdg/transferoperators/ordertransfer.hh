#ifndef DUNE_HPDG_DG_MULTIGRID_ORDER_TRANSFER_HH
#define DUNE_HPDG_DG_MULTIGRID_ORDER_TRANSFER_HH

#include <memory>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/istl/bdmatrix.hh>

#include <dune/matrix-vector/transformmatrix.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/transferoperators/dynamicordertransfer.hh>

/** \brief Galerkin restriction and prolongation for Discontinuous Galerkin stiffness matrices
 *
 * This class implements the Galerkin restriciton and prolongation between DG bases of different orders on the same grid.
 * Matrices and vectors have to be blocked element-wise(!). This leads to very efficient computations since the transfer matrices are
 * block diagonal.
 */
namespace Dune {
  namespace HPDG {
    template<class VectorType, int dim>
    class DGOrderTransfer
    {


      using DynamicMatrixType = DynamicBCRSMatrix<typename VectorType::field_type>;
      using MatrixType = typename DynamicMatrixType::Matrix;

      using BlockType = typename MatrixType::block_type;
    public:

      using field_type = typename VectorType::field_type;

      using TransferOperatorType = DynamicMatrixType;

      virtual ~DGOrderTransfer() {}

      void setup(const DynamicMatrixType& fineReference, int maxOrder)
      {
        // set up index set (block diagonal):
        {
          MatrixIndexSet idx(fineReference.N(), fineReference.N());
          for (size_t i = 0; i < fineReference.N(); i++)
            idx.add(i,i);
          idx.exportIdx(matrix_.matrix());
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
          auto& transferBlock = matrix_.matrix()[i][i];
          auto blocks = fineReference.blockRows(i);
          // case 1: block is not too large, we can put an identity
          if (blocks<=maxBlockSize) {
            DUNE_THROW(Dune::Exception, "HERES STILL A BUG"); // TODO: die schleifen unten sind falsch. die matrizen sind nicht immer blocks x blocks
            for (size_t i = 0; i < blocks; i++)
              for (size_t j = 0; j < blocks; j++)
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
        coarseVector = makeDynamicBlockVector(matrix_);

        matrix_.matrix().mtv(fineVector, coarseVector); // coarseVector = matrix_^T * fineVector;
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
        matrix_.matrix().mv(coarseVector, fineVector); // fineVector = matrix_*coarseVector;
      }

      /** \brief Galerkin assemble a coarse stiffness matrix
      */
      void galerkinRestrict(const DynamicMatrixType& fineMatrix, DynamicMatrixType& coarseMatrix) const
      {

        const auto& fineMat = fineMatrix.matrix();
        auto& coarseMat = coarseMatrix.matrix();
        coarseMat = 0;

        // We do not directly call the multipl. method on the global matrices since we want to use the block diagonal structure
        // to avoid expensive exists() calls

        // Loop over nonzero entries of the fine matrix
        for (size_t r=0; r <fineMat.N(); r++) {
          const auto& row = fineMat[r];
          auto cIt = row.begin();
          const auto& cEnd = row.end();
          for (; cIt!=cEnd; ++cIt) {
            auto c = cIt.index(); // we're at index (r,c)
            // R^T *A * R (blockwise multiplication as R is block diagonal)
            Dune::MatrixVector::addTransformedMatrix(coarseMat[r][c], matrix_.matrix()[r][r], fineMat[r][c], matrix_.matrix()[c][c]);
          }
        }
      }

      /** \brief Set Occupation of Galerkin restricted coarse stiffness matrix
    *
    * Set occupation of Galerkin restricted coarse matrix. Call this one before
    * galerkinRestrict to ensure all non-zeroes are present
    * \param fineMat The fine level matrix
    * \param coarseMat The coarse level matrix
    */
      void galerkinRestrictSetOccupation(const DynamicMatrixType& fineMat, DynamicMatrixType& coarseMat) const
      {
        // This one is easy, as the transfer operator has block diag. structure. Hence,
        // the coarse matrix will have the same block structure as the fine matrix.

        Dune::MatrixIndexSet idx(fineMat.N(), fineMat.M());
        idx.import(fineMat.matrix());
        idx.exportIdx(coarseMat.matrix());

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
