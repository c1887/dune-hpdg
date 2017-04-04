#ifndef DUNE_HPDG_DG_MULTIGRID_TRANSFER_HH
#define DUNE_HPDG_DG_MULTIGRID_TRANSFER_HH

#include <memory>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/istl/bdmatrix.hh>

#include <dune/matrix-vector/transformmatrix.hh>

/** \brief Galerkin restriction and prolongation for Discontinuous Galerkin stiffness matrices
 *
 * This class implements the Galerkin restriciton and prolongation between DG bases of different orders on the same grid.
 * Matrices and vectors have to be blocked element-wise(!). This leads to very efficient computations since the transfer matrices are
 * block diagonal.
 */
namespace Dune {
  namespace HPDG {
    template<
        class VectorType,
        int coarseBlocksize,
        class BitVectorType = Dune::BitSetVector<VectorType::block_type::dimension>,
        class MatrixType = Dune::BCRSMatrix< typename Dune::FieldMatrix<
                                               typename VectorType::field_type, VectorType::block_type::dimension, VectorType::block_type::dimension> > >
    class DGMultigridTransfer
    {


      typedef typename VectorType::field_type field_type;

    public:

      enum {blocksize = VectorType::block_type::dimension};
      enum {coarseBlock = coarseBlocksize};
      typedef Dune::BDMatrix< Dune::FieldMatrix< field_type, blocksize, coarseBlocksize> > TransferOperatorType;

      // Ctor that already knows the number of elements (and hence the number of block rows)
      DGMultigridTransfer(int nElements) {
        matrix_ = TransferOperatorType(nElements);
      }

      virtual ~DGMultigridTransfer() {}


      template <typename SetupFunction>
      void setup(SetupFunction&& assembleMatrix)
      {
        // This needs some dependencies not provided in dune-solvers, so we let the actual assemlby happen in an extern place
        assembleMatrix(matrix_);
      }

      template <typename SetupFunction>
      void setup(int nElements, SetupFunction&& assembleMatrix)
      {
        matrix_= TransferOperatorType(nElements);
        // This needs some dependencies not provided in dune-solvers, so we let the actual assemlby happen in an extern place
        assembleMatrix(matrix_);
      }

      /** \brief Restrict a function from the fine onto the coarse grid
     */
      template <typename CoarseVectorType>
      void restrict(const VectorType& fineVector, CoarseVectorType& coarseVector) const
      {
        matrix_.mtv(fineVector, coarseVector); // coarseVector = matrix_^T * fineVector;
      }

      /** \brief Prolong a function from the coarse onto the fine grid
     */
      template <typename CoarseVectorType>
      void prolong(const CoarseVectorType& coarseVector, VectorType& fineVector) const
      {
        matrix_.mv(coarseVector, fineVector); // fineVector = matrix_*coarseVector;
      }

      /** \brief Galerkin assemble a coarse stiffness matrix
     */
      template <typename CoarseMatrixType>
      void galerkinRestrict(const MatrixType& fineMat, CoarseMatrixType& coarseMat) const
      {
        coarseMat = 0;

        // We do not directly call the multipl. method on the global matrices since we want to use the block diagonal structure
        // to avoid expensive exists() calls

        // Loop over nonzero entries of the fine matrix
        for (size_t r=0; r <fineMat.N(); r++) {
          const auto& row = fineMat[r];
          auto cIt = row.begin();
          auto cEnd = row.end();
          for (; cIt!=cEnd; ++cIt) {
            auto c = cIt.index(); // we're at index (r,c)
            // R^T *A * R (blockwise multiplication as R is block diagonal)
            Dune::MatrixVector::addTransformedMatrix(coarseMat[r][c], matrix_[r][r], fineMat[r][c], matrix_[c][c]);
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
      template <typename CoarseMatrixType>
      void galerkinRestrictSetOccupation(const MatrixType& fineMat, CoarseMatrixType& coarseMat) const
      {
        // This one is easy, as the transfer operator has block diag. structure. Hence,
        // the coarse matrix will have the same block structure as the fine matrix.

        Dune::MatrixIndexSet idx(fineMat.N(), fineMat.M());

        // Copy indices from the fine matrix into the coarse matrix
        for (size_t i = 0; i<fineMat.N(); ++i) {
          const auto& row = fineMat[i];

          auto it = row.begin();
          auto end = row.end();
          for (; it!=end; ++it) {
            auto j = it.index();
            idx.add(i, j);
          }
        }
        idx.exportIdx(coarseMat);
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
