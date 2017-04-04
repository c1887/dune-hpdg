#ifndef DUNE_HPDG_DG_Q1_TO_Q1_HH
#define DUNE_HPDG_DG_Q1_TO_Q1_HH

#include <memory>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/istl/bdmatrix.hh>
#include <dune/solvers/common/algorithm.hh>

#include <dune/matrix-vector/transformmatrix.hh>

/** \brief Galerkin restriction and prolongation for blocked matrices
 * Essentially the same as DGMultigridTransfer but this does not assume that we have a block-diagonal transfer operator
 *
 */
namespace Dune {
  namespace HPDG{
    template<
        class VectorType,
        int coarseBlocksize, // TODO brauche ich den noch?
        class BitVectorType = Dune::BitSetVector<VectorType::block_type::dimension>,
        class MatrixType = Dune::BCRSMatrix< typename Dune::FieldMatrix<
                                               typename VectorType::field_type, VectorType::block_type::dimension, VectorType::block_type::dimension> > >
    class BlockTransferOperator
    {
      typedef typename VectorType::field_type field_type;

    public:

      enum {blocksize = VectorType::block_type::dimension};
      enum {coarseBlock = coarseBlocksize};
      using TransferOperatorType = Dune::BCRSMatrix< Dune::FieldMatrix< field_type, blocksize, coarseBlocksize> >;

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
        coarseVector.resize(matrix_.M());
        matrix_.mtv(fineVector, coarseVector); // coarseVector = matrix_^T * fineVector;
      }

      /** \brief Prolong a function from the coarse onto the fine grid
     */
      template <typename CoarseVectorType>
      void prolong(const CoarseVectorType& coarseVector, VectorType& fineVector) const
      {
        fineVector.resize(matrix_.N());
        matrix_.mv(coarseVector, fineVector); // fineVector = matrix_*coarseVector;
      }

      /** \brief Galerkin assemble a coarse stiffness matrix
     */
      template <typename CoarseMatrixType>
      void galerkinRestrict(const MatrixType& fineMat, CoarseMatrixType& coarseMat) const
      {
        coarseMat = 0;
        for (std::size_t i =0; i< fineMat.N(); i++) {
          const auto& Ai = fineMat[i];
          Dune::Solvers::sparseRangeFor(Ai, [&](auto&& Aij, auto&& j) {
            const auto& Ti = matrix_[i];
            const auto& Tj = matrix_[j];
            Dune::Solvers::sparseRangeFor(Ti, [&](auto&& Tik, auto&& k) {
              Dune::Solvers::sparseRangeFor(Tj, [&](auto&& Tjl, auto&& l) {
                  Dune::MatrixVector::addTransformedMatrix(coarseMat[k][l], Tik, Aij, Tjl);
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
    * \param setup The function that sets the index set
    */
      template <typename CoarseMatrixType>
      void galerkinRestrictSetOccupation(const MatrixType& fineMat, CoarseMatrixType& coarseMat) const
      {
        auto m = matrix_.M(); // #coarse functions
        auto indices = Dune::MatrixIndexSet(m, m);
        for (std::size_t i =0; i< fineMat.N(); i++) {
          const auto& Ai = fineMat[i];
          Dune::Solvers::sparseRangeFor(Ai, [&](auto&& Aij, auto&& j) {
            const auto& Ti = matrix_[i];
            const auto& Tj = matrix_[j];
            Dune::Solvers::sparseRangeFor(Ti, [&](auto&& Tik, auto&& k) {
              Dune::Solvers::sparseRangeFor(Tj, [&](auto&& Tjl, auto&& l) {
                indices.add(k, l);
              });
            });
          });
        }
        indices.exportIdx(coarseMat);
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
