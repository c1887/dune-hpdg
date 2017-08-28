// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_DYNAMIC_BLOCKVECTOR_HH
#define DUNE_HPDG_DYNAMIC_BLOCKVECTOR_HH

#include <vector>
#include <memory>

#include <dune/hpdg/common/vectorwindow.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>

namespace Dune {
  namespace HPDG {

    template<class K>
    class DynamicBlockVector {
      using BlockType = Dune::HPDG::VectorWindow<K>;
      enum {blocklevel = BlockType::blocklevel +1 };

      public:

      using field_type = K;
      using size_type = size_t;
      using block_type = BlockType;

      DynamicBlockVector(size_t n, size_t blockRows=1) :
        n_(n),
        rowMap_(n),
        vector_(n)
      {for (auto& e: rowMap_) e = blockRows;}

      DynamicBlockVector() :
        n_(0),
        rowMap_(0),
        vector_(0) {}

      DynamicBlockVector(const DynamicBlockVector& other) {
        n_ = other.n_;
        rowMap_ = other.rowMap_;
        vector_.resize(n_);
        update();
        for (size_t i = 0; i < size_; i++)
          data_[i] = other.data_[i];
      }

      /** Set the size of the DynamicBlockVector.
       *
       * \warning: This does _not_ allocate the memory!
       */
      void setSize(size_t n) {
        n_=n;
        vector_.resize(n_);
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

      /** (Re-)allocates memory and sets pointers of the matrix windows.
       *
       * This is needed before actually using the matrix!
       */
      void update() {
        allocateMemory();
        resetBlocks();
      }

      const BlockType& operator[](size_t i) const {
        return vector_[i];
      }

      BlockType& operator[](size_t i) {
        return vector_[i];
      }

      auto size() const {
        return n_;
      }
      auto N() const {
        return n_;
      }

      DynamicBlockVector& operator=(const field_type& scalar) {
        for (size_t i = 0; i < size_; ++i)
          data_[i]=scalar;
        return *this;
      }

      DynamicBlockVector& operator=(const DynamicBlockVector& other) {
        n_ = other.n_;
        rowMap_ = other.rowMap_;
        vector_.resize(n_);
        update();
        for (size_t i = 0; i < size_; i++)
          data_[i] = other.data_[i];

        return *this;
      }

      DynamicBlockVector& operator+=(const DynamicBlockVector& other) {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        for (size_t i = 0; i < size_; i++)
          data_[i] += other.data_[i];
        return *this;
      }

      DynamicBlockVector& operator-=(const DynamicBlockVector& other) {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        for (size_t i = 0; i < size_; i++)
          data_[i] -= other.data_[i];
        return *this;
      }

      DynamicBlockVector& operator*=(const field_type& k) {
        for (size_t i = 0; i < size_; i++)
          data_[i] *= k;
        return *this;
      }

      DynamicBlockVector& operator/=(const field_type& k) {
        for (size_t i = 0; i < size_; i++)
          data_[i] /= k;
        return *this;
      }

      DynamicBlockVector operator+(const DynamicBlockVector& other) const {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        auto v=*this;
        v+=other;
        return v;
      }

      DynamicBlockVector operator-(const DynamicBlockVector& other) const {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        auto v=*this;
        v-=other;
        return v;
      }

      DynamicBlockVector& axpy(const field_type& a, const DynamicBlockVector& other) {
        for (size_t i = 0; i < size_; i++)
          data_[i]+=(a*other.data_[i]);
        return *this;
      }

      // scalar product
      field_type operator*(const DynamicBlockVector& other) {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        field_type sum =0;
        for (size_t i = 0; i < size_; i++)
          sum+=data_[i]*other.data_[i];
        return sum;
      }

      auto begin() {
        return vector_.begin();
      }

      auto end() {
        return vector_.end();
      }

      auto begin() const {
        return vector_.cbegin();
      }

      auto end() const {
        return vector_.cend();
      }

      auto dimension() const {
        return size_;
      }

      // give STL-like access
      auto& at(size_t i) {
        DUNE_ASSERT_BOUNDS(i<n_);
        return vector_[i];
      }
      const auto& at(size_t i) const {
        DUNE_ASSERT_BOUNDS(i<n_);
        return vector_[i];
      }
      auto& front() {
        return vector_.front();
      }
      const auto& front() const {
        return vector_.front();
      }
      auto& back() {
        return vector_.back();
      }
      const auto& back() const {
        return vector_.back();
      }

      private:

      /** Allocates memory.
       *
       * If memory has been allocated before,
       * it will be released and new memory will be allocated!
       *
       */
      void allocateMemory() {
        const auto oldSize = size_;
        size_ = calculateSize();

        // if the old and new size match, do not allocate again
        if (size_ == oldSize)
          return;

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
          count += rowMap_[i];
        }
        return count;
      }

      /** Sets the pointers in the vector windows, aka. initializes the blocks */
      void resetBlocks() {
        auto* currentPtr = data_.get();
        for (size_t i = 0; i < n_; i++) {
          auto& Vi = vector_[i]; // get i-th block
          Vi.set(currentPtr, rowMap_[i]);
          currentPtr += rowMap_[i]; // ptr arithmetic, move to point after current block
        }
      }


      size_t n_;
      std::vector<size_t> rowMap_; // stores the number of rows each block row has
      std::vector<VectorWindow<K>> vector_;
      std::unique_ptr<K[]> data_;
      size_t size_;
    };

    /** Return a readily allocated dynamic BlockVector with the same blocking as a given DynamicBCRSMatrix M,
     * i.e. one such that M*v is valid.
     */
    template<class field_type>
    DynamicBlockVector<field_type> makeDynamicBlockVector(const DynamicBCRSMatrix<field_type>& matrix) {
      DynamicBlockVector<field_type> v;
      v.setSize(matrix.M());
      for (size_t i = 0; i < matrix.N(); i++)
        v.blockRows(i)=matrix.blockColumns(i);
      v.update();
      return v;
    }
  }
}
#endif//DUNE_HPDG_DYNAMIC_BLOCKVECTOR_HH
