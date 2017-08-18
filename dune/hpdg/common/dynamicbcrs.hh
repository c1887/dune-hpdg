// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_DYNAMIC_BCRS_HH
#define DUNE_HPDG_DYNAMIC_BCRS_HH

#include <vector>
#include <memory>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/hpdg/common/matrixwindow.hh>
#include <dune/matrix-vector/algorithm.hh>

namespace Dune {
  namespace HPDG {

    template<class K>
    class DynamicBCRSMatrix {
      using BlockType = Dune::HPDG::MatrixWindow<K>;
      using BCRS = Dune::BCRSMatrix<BlockType>;

      public:

      using Matrix = Dune::BCRSMatrix<BlockType>;
      using field_type = K;

      DynamicBCRSMatrix(size_t n, size_t m, size_t blockRows=1) :
        n_(n),
        m_(m),
        rowMap_(n)
      {for (auto& e: rowMap_) e = blockRows;}

      DynamicBCRSMatrix() :
        n_(0),
        m_(0),
        rowMap_(0) {}

      /** Get the managed BCRS matrix in Dune::BCRSMatrix format*/
      BCRS& matrix() {
        return matrix_;
      }

      /** Get the managed BCRS matrix in Dune::BCRSMatrix format*/
      const BCRS& matrix() const {
        return matrix_;
      }

      /** Set the size of the DynamicBCRS matrix.
       * Note that this will _not_ modify the index set
       * of the Dune::BCRSMatrix member!
       */
      void setSize(size_t n, size_t m) {
        n_=n;
        m_=m;
        rowMap_.resize(n_);
      }

      /** Get the number of rows in the i-th block row */
      size_t& blockRows(size_t i) {
        DUNE_ASSERT_BOUNDS(i<n_);
        return rowMap_[i];
      }

      /** Get the number of rows in the i-th block row */
      const size_t& blockRows(size_t i) const {
        DUNE_ASSERT_BOUNDS(i<n_);
        return rowMap_[i];
      }

      /** Updates the matrix size after the index set was set externally
       */
      void finishIdx() {
        setSize(matrix_.N(), matrix_.M());
      }

      /** (Re-)allocates memory and sets pointers of the matrix windows. 
       *
       * This is needed before actually using the matrix!
       */
      void update() {
        allocateMemory();
        resetBlocks();
      }

      private:

      /** Allocates memory.
       *
       * If memory has been allocated before, 
       * it will be released and new memory will be allocated!
       */
      void allocateMemory() {
        auto size = calculateSize();
        if (data_ != nullptr) {
          data_.reset(new field_type[size]);
        }
        else {
          data_ = std::make_unique<field_type[]>(size);
        }
      }

      /** Calculate the amount of memory needed*/
      size_t calculateSize() const {
        size_t count = 0;
        for (size_t i = 0; i < n_; i++) {
          auto& Qi = matrix_[i];
          MatrixVector::sparseRangeFor(Qi, [&](auto&&, auto&&j) {
              count+=rowMap_[i]*rowMap_[j];
              });
        }
        return count;
      }

      /** Sets the pointers in the matrix windows, aka. initializes the blocks */
      void resetBlocks() {
        auto currentPtr = data_.get();
        for (size_t i = 0; i < n_; i++) {
          auto& Qi = matrix_[i];
          MatrixVector::sparseRangeFor(Qi, [&](auto&& Qij, auto&&j) {
              Qij.set(currentPtr, rowMap_[i], rowMap_[j]);
              currentPtr += rowMap_[i]*rowMap_[j]; // ptr arithmetic, move to point after current block
              });
        }
      }


      size_t n_;
      size_t m_;
      std::vector<size_t> rowMap_; // stores the number of rows each block row has
      BCRS matrix_;
      std::unique_ptr<field_type[]> data_;
    };
  }
}
#endif//DUNE_HPDG_DYNAMIC_BCRS_HH
