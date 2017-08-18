#ifndef DUNE_HPDG_DYNAMIC_ORDER_TRANSFER_HH
#define DUNE_HPDG_DYNAMIC_ORDER_TRANSFER_HH

#include <memory>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/istl/bdmatrix.hh>
#include <dune/istl/scaledidmatrix.hh>

#include <dune/matrix-vector/transformmatrix.hh>
#include <dune/matrix-vector/algorithm.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/localfunctions/lagrange/qkglfactory.hh>

#include <dune/hpdg/common/blockwiseoperations.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper

namespace Dune {
namespace HPDG {

namespace Impl {
  template<class MatrixType, int dim>
  struct TransferMatrixCache {

    const MatrixType& operator()(int newOrder, int currentOrder) {
      auto pair = std::pair<int, int> (newOrder, currentOrder);
      auto mat = matrix_.find(pair);
      if (mat == matrix_.end()) {
        // compute matrix, put it in cache, return it.
        matrix_[pair]= makeTransferMatrix(cache_.get(newOrder), cache_.get(currentOrder));
        return matrix_[pair];
      }
      else return mat->second;
    }

    private:

    using field_type = typename MatrixType::field_type; std::map<std::pair<int, int>, MatrixType> matrix_;
    using FECache = Dune::DynamicOrderQkGLLocalFiniteElementCache<field_type, field_type, dim>;
    FECache cache_;

    using FE = typename FECache::FiniteElementType;
    MatrixType makeTransferMatrix(const FE &fineFE, const FE &coarseFE) {
      auto numCoarse = coarseFE.size();
      auto numFine = fineFE.size();

      auto matrix = MatrixType(numFine, numCoarse);
      std::cout << "matrix: " << matrix.N() << " x " << matrix.M() << std::endl;

      std::vector<typename FE::Traits::LocalBasisType::Traits::RangeType> values(
          numCoarse);

      using FunctionBaseClass =
        typename Dune::LocalFiniteElementFunctionBase<FE>::type;
      using LocalBasisWrapper = LocalBasisComponentWrapper<
        typename FE::Traits::LocalBasisType, FunctionBaseClass>;
      LocalBasisWrapper coarseBasisFunction(coarseFE.localBasis(), 0);

      for (size_t j = 0; j < numCoarse; j++) {
        /* Interpolate values of the j-th coarse function*/
        coarseBasisFunction.setIndex(j);
        fineFE.localInterpolation().interpolate(coarseBasisFunction, values);

        /* copy them into the local block */
        for (size_t i = 0; i < numFine; i++) {
          matrix[i][j] = (field_type) values[i];
        }
      }
      return matrix;
    }

  };

  template<class MatrixType>
  struct IdentityCache {

    const MatrixType& operator()(size_t i) {
      auto mat = matrix_.find(i);
      if (mat==matrix_.end()) {
        matrix_[i] = MatrixType(i,i);

        // set id
        for (size_t k = 0; k < i; k++)
          for (size_t l = 0; l < i; l++)
            matrix_[i][k][l] = (k==l) ? 1 : 0;

        return matrix_[i];
      }
      else return mat->second;
    }

    private:

    using field_type = typename MatrixType::field_type;
    std::map<size_t, MatrixType> matrix_;

  };

}

template <class VectorType, int dim>
class DynamicOrderTransfer {
  public:

  using field_type = typename VectorType::field_type;
  using MatrixType = Dune::HPDG::DynamicBCRSMatrix<field_type>;

  /** \brief Restrict a function from the fine onto the coarse grid
  */
  //template <typename CoarseVectorType>
  //void restrict(const VectorType &fineVector,
                //CoarseVectorType &coarseVector) const {
    //Dune::HPDG::scaleByTransposedBlock(matrix_, fineVector, coarseVector);
  //}

  /** \brief Prolong a function from the coarse onto the fine grid
  */
  //template <typename CoarseVectorType>
  //void prolong(const CoarseVectorType &coarseVector,
               //VectorType &fineVector) const {
    //Dune::HPDG::scaleByBlock(matrix_, coarseVector, fineVector);
  //}

  /** \brief Galerkin assemble a coarse stiffness matrix. All orders higher than input maxOrder will be restricted to
   * maxOrder. Lower orders will not be affected
   */
  void galerkinRestrict(const MatrixType &fineMat, MatrixType &coarseMat, int maxOrder) const {
    coarseMat.matrix() = 0;

    // the idea is the following: the global transfer matrix is block diagonal.
    // We truncate all orders that are greater than maxorder, hence, the corresponding blocks hold the transfer information for an element.
    // On all others, there is an identity blocks, where we don't do anything.
    // Instead of computing the global transfer, we only compute the local blocks once (in the matrix_ cache)
    //
    auto maxBlockSize = orderToBlockSize(maxOrder);

    for (std::size_t i = 0; i < coarseMat.matrix().N(); i++) {
      auto &Ci = coarseMat.matrix()[i]; // current row

      const TransferType* rowMatrix;
      const TransferType* colMatrix;
      auto rowIsIdentity = true; // checked if row element has lesser than maxOrder degree

      if (fineMat.blockRows(i)>maxBlockSize) {
        rowIsIdentity = false;
        rowMatrix = &matrix_(blockSizeToOrder(fineMat.blockRows(i)), maxOrder);
      }
      else
        rowMatrix = &identity_(fineMat.blockRows(i));

      // S^T A S
      Dune::MatrixVector::sparseRangeFor(Ci, [&](auto &&Cij, auto &&j) {
        if (fineMat.blockRows(j)<=maxBlockSize and rowIsIdentity) {
          Cij=fineMat.matrix()[i][j]; // == I^T*fineMat[i][j]*I , but I do not want to compute explitcitly here
        }
        else {
          if (fineMat.blockRows(j)>maxBlockSize) {
            colMatrix = &matrix_(this->blockSizeToOrder(fineMat.blockRows(j)), maxOrder);
          }
          else
            rowMatrix = &identity_(fineMat.blockRows(j));

          Dune::MatrixVector::addTransformedMatrix(Cij, *rowMatrix, fineMat.matrix()[i][j], *colMatrix);
        }
      });
    }
  }

  /** \brief Set Occupation of Galerkin restricted coarse stiffness matrix
   *
   * Set occupation of Galerkin restricted coarse matrix. Call this one before
   * galerkinRestrict to ensure all non-zeroes are present
   * \param fineMat The fine level matrix
   * \param coarseMat The coarse level matrix
   * \param order The maximal order elements may have. All greater orders will be restricted to order.
   */
  void galerkinRestrictSetOccupation(const MatrixType &fineMat, MatrixType &coarseMat, int order) const {
    // This one is easy, as the transfer operator has block diag. structure.
    // Hence, the coarse matrix will have the same block structure as the fine matrix.
    Dune::MatrixIndexSet idx(fineMat.matrix().N(), fineMat.matrix().M());

    // Copy indices from the fine matrix into the coarse matrix
    idx.import(fineMat.matrix());
    idx.exportIdx(coarseMat.matrix());

    //finish the setup of the coarseMat
    coarseMat.finishIdx();

    // set blockrow's rows: if greater than maxSize, truncate to maxSize, else copy from fineMat
    auto maxSize= orderToBlockSize(order);
    for (size_t i = 0; i < coarseMat.matrix().N(); i++)
      coarseMat.blockRows(i) = (fineMat.blockRows(i) > maxSize) ? maxSize : fineMat.blockRows(i);

    // finish setup
    coarseMat.update();
  }

private:
  int blockSizeToOrder(const size_t& blockSize) const {
    auto val = std::pow(blockSize, 1.0/dim);
    return (int) val-1;
  }
  size_t orderToBlockSize(const int& order) const {
    return (size_t) std::pow(order+1, dim);
  }

  using TransferType = Dune::DynamicMatrix<double>; // TODO: Think about proper matrix type
  mutable Impl::TransferMatrixCache<TransferType, dim> matrix_;
  mutable Impl::IdentityCache<TransferType> identity_;


};
} // end namespace HPDG
} // end namespace Dune

#endif
