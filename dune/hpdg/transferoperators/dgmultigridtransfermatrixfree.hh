#ifndef DUNE_HPDG_DG_MULTIGRID_TRANSFER_MATRIX_FREE_HH
#define DUNE_HPDG_DG_MULTIGRID_TRANSFER_MATRIX_FREE_HH

#include <memory>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/bitsetvector.hh>
#include <dune/istl/bdmatrix.hh>

#include <dune/matrix-vector/transformmatrix.hh>
#include <dune/solvers/common/algorithm.hh>

#include <dune/hpdg/common/blockwiseoperations.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper

/** \brief Galerkin restriction and prolongation for Discontinuous Galerkin
 *stiffness matrices
 *
 * \todo doc me!
 */
namespace Dune {
namespace HPDG {
template <
    class VectorType, int coarseBlocksize,
    class BitVectorType = Dune::BitSetVector<VectorType::block_type::dimension>,
    class MatrixType = Dune::BCRSMatrix<typename Dune::FieldMatrix<
        typename VectorType::field_type, VectorType::block_type::dimension,
        VectorType::block_type::dimension>>>
class DGMultigridTransferMatrixFree {

  typedef typename VectorType::field_type field_type;

public:
  enum { blocksize = VectorType::block_type::dimension };
  enum { coarseBlock = coarseBlocksize };
  using TransferOperatorType =
      Dune::BDMatrix<Dune::FieldMatrix<field_type, blocksize, coarseBlocksize>>;
  using BlockType = typename TransferOperatorType::block_type;

  template <typename SetupFunction> void setup(SetupFunction &&assembleMatrix) {
    assembleMatrix(matrix_);
  }

  /* Get polynomial orders from the class templates*/
  template <class CFE, class FFE>
  void setup(const CFE &coarseFE, const FFE &fineFE) {
    auto numCoarse = coarseFE.size();
    auto numFine = fineFE.size();
    assert(numCoarse == matrix_.M());
    assert(numFine == matrix_.N());

    std::vector<typename CFE::Traits::LocalBasisType::Traits::RangeType> values(
        numCoarse);

    using FunctionBaseClass =
        typename Dune::LocalFiniteElementFunctionBase<CFE>::type;
    using LocalBasisWrapper = LocalBasisComponentWrapper<
        typename CFE::Traits::LocalBasisType, FunctionBaseClass>;
    LocalBasisWrapper coarseBasisFunction(coarseFE.localBasis(), 0);

    for (size_t j = 0; j < numCoarse; j++) {
      /* Interpolate values of the j-th coarse function*/
      coarseBasisFunction.setIndex(j);
      fineFE.localInterpolation().interpolate(coarseBasisFunction, values);

      /* copy them into the local block */
      // auto localCoarse = coarseIndexSet.index(j)[1];
      for (size_t i = 0; i < numFine; i++) {
        // auto localFine = fineIndexSet.index(i)[1];
        matrix_[i][j] = values[i];
      }
    }
  }

  template <typename SetupFunction>
  void setup(int nElements, SetupFunction &&assembleMatrix) {
    // This needs some dependencies not provided in dune-solvers, so we let the
    // actual assemlby happen in an extern place
    assembleMatrix(matrix_);
  }

  /** \brief Restrict a function from the fine onto the coarse grid
 */
  template <typename CoarseVectorType>
  void restrict(const VectorType &fineVector,
                CoarseVectorType &coarseVector) const {
    Dune::HPDG::scaleByTransposedBlock(matrix_, fineVector, coarseVector);
  }

  /** \brief Prolong a function from the coarse onto the fine grid
 */
  template <typename CoarseVectorType>
  void prolong(const CoarseVectorType &coarseVector,
               VectorType &fineVector) const {
    Dune::HPDG::scaleByBlock(matrix_, coarseVector, fineVector);
  }

  /** \brief Galerkin assemble a coarse stiffness matrix
 */
  template <typename CoarseMatrixType>
  void galerkinRestrict(const MatrixType &fineMat,
                        CoarseMatrixType &coarseMat) const {
    coarseMat = 0;
    for (std::size_t i = 0; i < coarseMat.N(); i++) {
      auto &Ci = coarseMat[i];
      Dune::Solvers::sparseRangeFor(Ci, [&](auto &&Cij, auto &&j) {
        Dune::MatrixVector::addTransformedMatrix(Cij, matrix_, fineMat[i][j],
                                                 matrix_);
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
  template <typename CoarseMatrixType>
  void galerkinRestrictSetOccupation(const MatrixType &fineMat,
                                     CoarseMatrixType &coarseMat) const {
    // This one is easy, as the transfer operator has block diag. structure.
    // Hence,
    // the coarse matrix will have the same block structure as the fine matrix.

    Dune::MatrixIndexSet idx(fineMat.N(), fineMat.M());

    // Copy indices from the fine matrix into the coarse matrix
    for (size_t i = 0; i < fineMat.N(); ++i) {
      const auto &row = fineMat[i];

      auto it = row.begin();
      auto end = row.end();
      for (; it != end; ++it) {
        auto j = it.index();
        idx.add(i, j);
      }
    }
    idx.exportIdx(coarseMat);
  }

  /** \brief Direct access to the operator matrix, if you absolutely want it! */
  const auto &getMatrix() const { return matrix_; }

protected:
  BlockType matrix_;
};
} // end namespace Solvers
} // end namespace Dune

#endif
