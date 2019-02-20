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
    class DynamicBCRSMatrix
    : public BCRSMatrix<MatrixWindow<K>>
    {
      using BlockType = Dune::HPDG::MatrixWindow<K>;

      public:

      using Base = Dune::BCRSMatrix<BlockType>;

      DynamicBCRSMatrix(size_t n, size_t m, size_t blockRows=1) :
        Base(n,m),
        rowMap_(n),
        colMap_(m)
      {
        for (auto& e: rowMap_) e = blockRows;
        for (auto& e: colMap_) e = blockRows;
      }

      DynamicBCRSMatrix() :
        Base(),
        rowMap_(0),
        colMap_(0) {}

      DynamicBCRSMatrix(const DynamicBCRSMatrix& other) {
        dynamic_cast<Base&>(*this) = dynamic_cast<const Base&>(other);
        rowMap_= other.rowMap_;
        colMap_= other.colMap_;
        {
        }
        update();

        // copy data from one C-array to the other
        K* data = entries_.get();
        K* otherdata = other.entries_.get();
        for (size_t i = 0; i < size_; i++)
          *data++ = *otherdata++;
      }

      DynamicBCRSMatrix(DynamicBCRSMatrix&& tmp) {
        swapDynamicParts(tmp);
        dynamic_cast<Base&>(*this) = dynamic_cast<Base&&>(tmp);
        resetBlocks();
      }

      DynamicBCRSMatrix& operator=(DynamicBCRSMatrix&& tmp) {
        swapDynamicParts(tmp);
        dynamic_cast<Base&>(*this) = dynamic_cast<Base&&>(tmp);
        resetBlocks();
        return *this;
      }


      /** Set the size of the DynamicBCRS matrix.*/
      void setSize(size_t n, size_t m) {
        Base::setSize(n,m);
        rowMap_.resize(n);
        colMap_.resize(m);
      }

      // TODO: Integrate this in a more transparent manner.
      /** If your index set was assembled through a casted BCRSMatrix
       * version of this, the vectors containing the sizes of the blocks and columns
       * will not be resized. This is done via this method.*/
      void finishIdx() {
        rowMap_.resize(this->n);
        colMap_.resize(this->m);
      }

      DynamicBCRSMatrix& operator=(const K& scalar) {
        for(std::size_t i = 0; i < size_; i++) {
          entries_[i]=scalar;
        }
        return *this;
      }

      /** return this dynamically casted to BCRSMatrix<..>
       * This is useful if a method explicitly expects a BCRSMatrix object
       */
      Base& asBCRSMatrix() {
        return dynamic_cast<Base&>(*this);
      }

      /** return this dynamically casted to BCRSMatrix<..>
       * This is useful if a method explicitly expects a BCRSMatrix object
       */
      const Base& asBCRSMatrix() const {
        return dynamic_cast<const Base&>(*this);
      }

      /** Get the number of rows in the i-th block row */
      auto& blockRows(size_t i) {
        DUNE_ASSERT_BOUNDS(i<this->n);
        return rowMap_[i];
      }

      /** Get the number of rows in the i-th block row */
      const auto& blockRows(size_t i) const {
        DUNE_ASSERT_BOUNDS(i<this->n);
        return rowMap_[i];
      }

      /** Get the number of columns in the i-th block columns */
      auto& blockColumns(size_t i) {
        DUNE_ASSERT_BOUNDS(i<this->m);
        return colMap_[i];
      }

      /** Get the number of columns in the i-th block columns */
      const auto& blockColumns(size_t i) const {
        DUNE_ASSERT_BOUNDS(i<this->m);
        return colMap_[i];
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

      DynamicBCRSMatrix& operator=(const DynamicBCRSMatrix& other) {
        dynamic_cast<Base&>(*this) = dynamic_cast<const Base&>(other);
        rowMap_= other.rowMap_;
        colMap_= other.colMap_;
        update();

        // copy data from one C-array to the other
        K* data = entries_.get();
        K* otherdata = other.entries_.get();
        for (size_t i = 0; i < size_; i++)
          *data++ = *otherdata++;

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
        if (entries_ != nullptr) {
          entries_.reset(new K[size_]);
        }
        else {
          entries_ = std::make_unique<K[]>(size_);
        }
      }

      /** Calculate the amount of memory needed*/
      size_t calculateSize() const {
        size_t count = 0;
        for (size_t i = 0; i < this->n; i++) {
          auto& Qi = (*this)[i];
          MatrixVector::sparseRangeFor(Qi, [&](auto&&, auto&&j) {
              count+=rowMap_[i]*colMap_[j];
              });
        }
        return count;
      }

      /** Sets the pointers in the matrix windows, aka. initializes the blocks */
      void resetBlocks() {
        auto* currentPtr = entries_.get();
        for (size_t i = 0; i < this->n; i++) {
          auto& Qi = (*this)[i];
          MatrixVector::sparseRangeFor(Qi, [&](auto&& Qij, auto&&j) {
              Qij.set(currentPtr, rowMap_[i], colMap_[j]);
              currentPtr += rowMap_[i]*colMap_[j]; // ptr arithmetic, move to point after current block
              });
        }
      }

      void swapDynamicParts(DynamicBCRSMatrix& tmp) {
        std::swap(rowMap_, tmp.rowMap_);
        std::swap(colMap_, tmp.colMap_);
        std::swap(entries_, tmp.entries_);
      }

      std::vector<size_t> rowMap_; // stores the number of rows each block row has
      std::vector<size_t> colMap_; // stores the number of rows each block row has
      std::size_t size_;
      // Contains the actual data for the matrix windows (which are stored in the BCRSMatrix member "data")
      std::unique_ptr<K[]> entries_; // TODO: Maybe use a std::vector instead.
    };
  }
}
#endif//DUNE_HPDG_DYNAMIC_BCRS_HH
