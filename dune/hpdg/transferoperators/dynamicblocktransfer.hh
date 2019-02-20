#ifndef DUNE_HPDG_DYNAMIC_BLOCK_TRANSFER_HH
#define DUNE_HPDG_DYNAMIC_BLOCK_TRANSFER_HH

#include <memory>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/matrix-vector/algorithm.hh>

#include <dune/matrix-vector/transformmatrix.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>

#include <dune/hpdg/transferoperators/ordertransfer.hh>
/** \brief Galerkin restriction and prolongation for blocked matrices
 * Essentially the same as DGMultigridTransfer but this does not assume that we have a block-diagonal transfer operator
 *
 */
// TODO: Ditch VectorType template. It's probably always DynamicBlocKVector
namespace Dune {
  namespace HPDG{
    template<class VectorType>
    class DynamicBlockTransferOperator
    {


      using DynamicMatrixType = DynamicBCRSMatrix<Dune::FieldMatrix<typename VectorType::field_type,1,1>>;
      using MatrixType = DynamicMatrixType;

      using BlockType = typename MatrixType::block_type;

    public:
      using field_type = typename VectorType::field_type;
      using TransferOperatorType = DynamicMatrixType;

      template <typename SetupFunction>
      void setup(SetupFunction&& assembleMatrix)
      {
        // This needs some dependencies not provided in dune-solvers, so we let the actual assemlby happen in an extern place
        assembleMatrix(matrix_);
      }


      /** \brief Restrict a function from the fine onto the coarse grid
     */
      template <typename CoarseVectorType>
      void restrict(const VectorType& fineVector, CoarseVectorType& coarseVector) const
      {
        coarseVector.setSize(matrix_->M());
        // step 1: prepare coarse vector:
        for (size_t i = 0; i < matrix_->M(); i++)
          coarseVector.blockRows(i)=matrix_->blockRows(i);
        coarseVector.update();

        // step 2: restrict
        matrix_->mtv(fineVector, coarseVector); // coarseVector = matrix_^T * fineVector;
 
      }

      /** \brief Prolong a function from the coarse onto the fine grid
     */
      template <typename CoarseVectorType>
      void prolong(const CoarseVectorType& coarseVector, VectorType& fineVector) const
      {
        fineVector.setSize(matrix_->N());
        for (size_t i = 0; i < matrix_->N(); i++) {
          fineVector.blockRows(i)=matrix_->blockRows(i);
        }
        fineVector.update();

        matrix_->mv(coarseVector, fineVector); // fineVector = matrix_*coarseVector;
      }

      /** \brief Galerkin assemble a coarse stiffness matrix
     */
      void galerkinRestrict(const DynamicMatrixType& fineMat, DynamicMatrixType& coarseMat) const
      {
        coarseMat = 0.;
        for (std::size_t i =0; i< fineMat.N(); i++) {
          const auto& Ai = fineMat[i];
          Dune::MatrixVector::sparseRangeFor(Ai, [&](auto&& Aij, auto&& j) {
            const auto& Ti = (*matrix_)[i];
            const auto& Tj = (*matrix_)[j];
            Dune::MatrixVector::sparseRangeFor(Ti, [&](auto&& Tik, auto&& k) {
              Dune::MatrixVector::sparseRangeFor(Tj, [&](auto&& Tjl, auto&& l) {
                  //Dune::MatrixVector::addTransformedMatrix(coarseMat.matrix()[k][l], Tik, Aij, Tjl);
                  Impl::addTransformedMatrix(coarseMat[k][l], Tik, Aij, Tjl);
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
      void galerkinRestrictSetOccupation(const DynamicMatrixType& fineMat, DynamicMatrixType& coarseMat) const
      {
        auto m = matrix_->M(); // #coarse functions
        auto indices = Dune::MatrixIndexSet(m, m);
        for (std::size_t i =0; i< fineMat.N(); i++) {
          const auto& Ai = fineMat[i];
          Dune::MatrixVector::sparseRangeFor(Ai, [&](auto&&, auto&& j) {
            const auto& Ti = (*matrix_)[i];
            const auto& Tj = (*matrix_)[j];
            Dune::MatrixVector::sparseRangeFor(Ti, [&](auto&&, auto&& k) {
              Dune::MatrixVector::sparseRangeFor(Tj, [&](auto&&, auto&& l) {
                indices.add(k, l);
              });
            });
          });
        }
        indices.exportIdx(coarseMat);

        // set sizes and allocate memory in the dynamic matrix
        coarseMat.finishIdx();

        for (size_t i = 0; i < m; i++) {
          coarseMat.blockRows(i)=coarseMat.blockColumns(i)=matrix_->blockColumns(i);
        }
        coarseMat.update();
      }

      /** \brief Direct access to the operator matrix, if you absolutely want it! */
      const TransferOperatorType& getMatrix() const {
        return *matrix_;
      }

    protected:

      std::shared_ptr<TransferOperatorType> matrix_;

    };
  } // end namespace HPDG
} // end namespace Dune

#endif
