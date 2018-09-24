// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_DYNAMIC_BCRS_HH
#define DUNE_HPDG_DYNAMIC_BCRS_HH

#include <vector>
#include <memory>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/hpdg/common/matrixwindow.hh>
#include <dune/matrix-vector/algorithm.hh>

namespace Dune {
  namespace HPDG {

    template<class K>
    class DynamicBCRSMatrix {

      public:

      using BlockType = Dune::HPDG::MatrixWindow<K>;
      using BCRS = Dune::BCRSMatrix<BlockType>;
      using Matrix = Dune::BCRSMatrix<BlockType>;
      using field_type = typename K::field_type;

      DynamicBCRSMatrix(size_t n, size_t m, size_t blockRows=1) :
        n_(n),
        m_(m),
        rowMap_(n),
        colMap_(m)
      {
        for (auto& e: rowMap_) e = blockRows;
        for (auto& e: colMap_) e = blockRows;
      }

      DynamicBCRSMatrix() :
        n_(0),
        m_(0),
        rowMap_(0),
        colMap_(0) {}

      DynamicBCRSMatrix(const DynamicBCRSMatrix& other) {
        n_=other.n_;
        m_=other.m_;
        rowMap_= other.rowMap_;
        colMap_= other.colMap_;
        {
          auto idx = Dune::MatrixIndexSet(n_, m_);
          idx.import(other.matrix_);
          idx.exportIdx(matrix_);
        }
        update();

        // copy data from one C-array to the other
        K* data = data_.get();
        K* otherdata = other.data_.get();
        for (size_t i = 0; i < size_; i++)
          *data++ = *otherdata++;
      }

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
        colMap_.resize(m_);
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

      /** Get the number of columns in the i-th block columns */
      size_t& blockColumns(size_t i) {
        DUNE_ASSERT_BOUNDS(i<n_);
        return colMap_[i];
      }

      /** Get the number of columns in the i-th block columns */
      const size_t& blockColumns(size_t i) const {
        DUNE_ASSERT_BOUNDS(i<n_);
        return colMap_[i];
      }
      /** Updates the matrix size after the index set was set externally
       */
      void finishIdx() {
        setSize(matrix_.N(), matrix_.M());
      }

      /** For convenience; sets the column sizes to the same values as the row sizes */
      void setSquare() {
        colMap_ = rowMap_;
      }

      /** (Re-)allocates memory and sets pointers of the matrix windows.
       *
       * This is needed before actually using the matrix!
       */
      void update() {
        allocateMemory();
        resetBlocks();
      }

      // get the look&feel of an actual BCRS matrix object

      auto N() const {
        return n_;
      }

      auto M() const {
        return m_;
      }

      template<class X, class Y>
      void mv(const X& x, Y& y) const {
        matrix_.mv(x,y);
      }

      template<class X, class Y>
      void mmv(const X& x, Y& y) const {
        matrix_.mmv(x,y);
      }

      template<class X, class Y>
      void mtv(const X& x, Y& y) const {
        matrix_.mtv(x,y);
      }

      template<class X, class Y>
      void umv(const X& x, Y& y) const {
        matrix_.umv(x,y);
      }

      template<class X, class Y>
      void umtv(const X& x, Y& y) const {
        matrix_.umtv(x,y);
      }

      auto& operator[](size_t i) {
        return matrix_[i];
      }

      const auto& operator[](size_t i) const {
        return matrix_[i];
      }

      DynamicBCRSMatrix& operator=(const DynamicBCRSMatrix& other) {
        n_=other.n_;
        m_=other.m_;
        rowMap_= other.rowMap_;
        colMap_= other.colMap_;
        {
          matrix_ = BCRS();
          auto idx = Dune::MatrixIndexSet(n_, m_);
          idx.import(other.matrix_);
          idx.exportIdx(matrix_);
        }
        update();

        // copy data from one C-array to the other
        K* data = data_.get();
        K* otherdata = other.data_.get();
        for (size_t i = 0; i < size_; i++)
          *data++ = *otherdata++;

        return *this;
      }

      DynamicBCRSMatrix& operator=(K scalar) {
        for (size_t i = 0; i < size_; i++)
          data_[i]=scalar;
        return *this;
      }


      private:

      /** Allocates memory.
       *
       * If memory has been allocated before,
       * it will be released and new memory will be allocated!
       */
      void allocateMemory() {
        size_ = calculateSize();
        if (data_ != nullptr) {
          data_.reset(new K[size_]);
        }
        else {
          data_ = std::make_unique<K[]>(size_);
        }
      }

      /** Calculate the amount of memory needed*/
      size_t calculateSize() const {
        size_t count = 0;
        for (size_t i = 0; i < n_; i++) {
          auto& Qi = matrix_[i];
          MatrixVector::sparseRangeFor(Qi, [&](auto&&, auto&&j) {
              count+=rowMap_[i]*colMap_[j];
              });
        }
        return count;
      }

      /** Sets the pointers in the matrix windows, aka. initializes the blocks */
      void resetBlocks() {
        auto* currentPtr = data_.get();
        for (size_t i = 0; i < n_; i++) {
          auto& Qi = matrix_[i];
          MatrixVector::sparseRangeFor(Qi, [&](auto&& Qij, auto&&j) {
              Qij.set(currentPtr, rowMap_[i], colMap_[j]);
              currentPtr += rowMap_[i]*colMap_[j]; // ptr arithmetic, move to point after current block
              });
        }
      }


      size_t n_;
      size_t m_;
      std::vector<size_t> rowMap_; // stores the number of rows each block row has
      std::vector<size_t> colMap_; // stores the number of rows each block row has
      BCRS matrix_;
      size_t size_;
      std::unique_ptr<K[]> data_;
    };
  }
}
#endif//DUNE_HPDG_DYNAMIC_BCRS_HH
