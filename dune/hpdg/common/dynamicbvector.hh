// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_DYNAMIC_BLOCKVECTOR_HH
#define DUNE_HPDG_DYNAMIC_BLOCKVECTOR_HH

#include <vector>
#include <memory>

#include <dune/common/fvector.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/parallel/communicator.hh>

#include <dune/hpdg/common/vectorwindow.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>

namespace Dune {
  namespace HPDG {

    namespace Impl {

      /** A wrapper that adds an index() method to a given
       * random access iterator */
      template<class Base, class BaseIter>
        class IndexedIterator : public BaseIter {
          public:
            template<class Iter>
              IndexedIterator (Base* b, Iter&& it) :
                BaseIter(std::forward<Iter>(it)),
                base_(b) {}

            auto index() const {
              return *this - base_->begin();
            }

          private:
            Base* base_;
        };

      template<class Base, class Iter>
        auto indexedIterator(Base& base, Iter&& iterator) {
          return IndexedIterator<Base, decltype(base.begin())>(&base, std::forward<Iter>(iterator));
        }
    }

    template<class K>
    class DynamicBlockVector {
      using BlockType = Dune::HPDG::VectorWindow<K>;

      public:

      enum {blocklevel = BlockType::blocklevel +1 };
      using field_type = typename K::field_type;
      using real_type = real_t<K>;
      using size_type = size_t;
      using block_type = BlockType;
      using value_type = K;

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

      /** Returns a vector of the same size and structure as the input but all
       * values are set to zero.
       *
       * This is basically a shorthand for
       *
       * auto foo = other;
       * foo = 0.;
       *
       * that also avoids the unnecessary copy of other's values.
       */
      static DynamicBlockVector createZeroVector(const DynamicBlockVector& other) {
        DynamicBlockVector zero_vector;
        zero_vector.n_ = other.n_;
        zero_vector.rowMap_ = other.rowMap_;
        zero_vector.vector_.resize(other.n_);
        zero_vector.update();
        for(size_t i = 0; i< zero_vector.size_; i++)
          zero_vector.data_[i] = 0;
        return zero_vector;
      }

      /** Returns a vector of the same size and structure as the input and allocated
       *  memory but with no value initialization of any kind.
       */
      static DynamicBlockVector uninitializedCopy(const DynamicBlockVector& other) {
        DynamicBlockVector zero_vector;
        zero_vector.n_ = other.n_;
        zero_vector.rowMap_ = other.rowMap_;
        zero_vector.vector_.resize(other.n_);
        zero_vector.update();
        return zero_vector;
      }

      /** Set the size of the DynamicBlockVector.
       *
       * \warning: This does _not_ allocate the memory!
       */
      void setSize(size_t n) {
        n_=n;

        // Reset the vector windows (i.e. set its length to zero
        // and the data ptr to nullptr. Moreover, release any self managed
        // memory (though there should not be any).
        // Otherwise, when the old entries will be copied
        // into the new one when resizing the vector_,
        // each VectorWindow will get his own data.
        // (This whole self managing design is kinda broken,
        // but there are too many places relying on it).
        for (auto& v: vector_) {
          v.reset();
        }
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

      /** Checks if any entries went rogue, i.e.
       * carry their own storage instead of being part of this
       * blockvector's memory.
       */
      bool checkValidity() const {

        for (const auto& vw: vector_) {
          if (vw.ownData_ != nullptr)
            return false;
        }

        return true;
      }

      const BlockType& operator[](size_t i) const {
        DUNE_ASSERT_BOUNDS(i<n_);
        return vector_[i];
      }

      BlockType& operator[](size_t i) {
        DUNE_ASSERT_BOUNDS(i<n_);
        return vector_[i];
      }

      auto size() const {
        return n_;
      }
      auto N() const {
        return n_;
      }

      template <typename T, typename = std::enable_if_t<Dune::IsNumber<T>::value>>
      DynamicBlockVector& operator=(const T& scalar) {
        for (size_t i = 0; i < size_; ++i)
          data_[i]=scalar;
        return *this;
      }

      DynamicBlockVector& operator=(const DynamicBlockVector& other) {
        n_ = other.n_;
        rowMap_ = other.rowMap_;

        // Reset the old entries, see comment in method 'setSize'.
        for (auto& v: vector_) {
          v.reset();
        }
        vector_.resize(n_);
        update();
        for (size_t i = 0; i < size_; i++)
          data_[i] = other.data_[i];

        return *this;
      }

      DynamicBlockVector& operator+=(const DynamicBlockVector& other) {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        for (size_t i = 0; i < n_; i++)
          vector_[i]+=other[i];
        return *this;
      }

      DynamicBlockVector& operator-=(const DynamicBlockVector& other) {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        for (size_t i = 0; i < n_; i++)
          vector_[i]-=other[i];
        return *this;
      }

      template <typename T, typename = std::enable_if_t<Dune::IsNumber<T>::value>>
      DynamicBlockVector& operator*=(const T& k) {
        for (size_t i = 0; i < size_; i++)
          data_[i] *= k;
        return *this;
      }

      template <typename T, typename = std::enable_if_t<Dune::IsNumber<T>::value>>
      DynamicBlockVector& operator/=(const T& k) {
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

      template <typename T, typename = std::enable_if_t<Dune::IsNumber<T>::value>>
      DynamicBlockVector& axpy(const T& a, const DynamicBlockVector& other) {
        for (size_t i = 0; i < size_; i++)
          data_[i]+=(a*other.data_[i]);
        return *this;
      }

      // scalar product
      field_type operator*(const DynamicBlockVector& other) const {
        DUNE_ASSERT_BOUNDS(other.n_==n_);
        field_type sum =0;
        for (size_t i = 0; i < size_; i++)
          sum+=data_[i]*other.data_[i];
        return sum;
      }

      auto begin() {
        return Impl::indexedIterator(vector_, vector_.begin());
      }

      auto end() {
        return Impl::indexedIterator(vector_, vector_.end());
      }

      auto begin() const {
        return Impl::indexedIterator(vector_, vector_.cbegin());
      }

      auto end() const {
        return Impl::indexedIterator(vector_, vector_.cend());
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

      auto two_norm() const {
        return std::sqrt((*this)*(*this));
      }

      auto two_norm2() const {
        return (*this)*(*this);
      }

      /** Construct a view to a vector's data.
       *
       * This view will be flat, i.e. it does not know about the blocking of
       * the dynamic vector. This might be handy, e.g., if the dynamic vector is
       * actually flat in the sense that all blocks are of size 1.
       */
      template<typename KK>
      friend auto flatVectorWindow(const DynamicBlockVector<KK>& v)
      {
        return VectorWindow<KK>(v.data_.get(), v.n_);
      }

      K* data() { return data_.get(); }

      const K* data() const { return data_.get(); }

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

        if (data_ != nullptr) {
          // if the old and new size match, do not allocate again
          if (size_ == oldSize)
            return;
          else
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
    template<class K>
    auto makeDynamicBlockVector(const DynamicBCRSMatrix<K>& matrix) {
      using field = Dune::FieldVector<typename K::field_type, K::cols>;
      DynamicBlockVector<field> v;
      v.setSize(matrix.M());
      for (size_t i = 0; i < matrix.N(); i++)
        v.blockRows(i)=matrix.blockColumns(i);
      v.update();
      return v;
    }

    /** TODO.
    template<class K>
    auto makeISTLVariableBlockVector(const DynamicBCRSMatrix<K>& matrix) {
    }
    */
  }

#if HAVE_MPI
  template<class K>
  struct CommPolicy<HPDG::DynamicBlockVector<K>> {
    using Type = HPDG::DynamicBlockVector<K>;

    using IndexedType = K;

    static const void* getAddress(const Type& v, int i) {
      return &(v[i])[0];
    }

    static int getSize(const Type& v, int index) {
      return v[index].size();
    }

  };
#endif

  template<class T>
  struct FieldTraits<HPDG::DynamicBlockVector<T>>
  {
    typedef typename HPDG::DynamicBlockVector<T>::field_type field_type;
    typedef typename HPDG::DynamicBlockVector<T>::real_type real_type;
  };
  template<class T>
  struct FieldTraits<const HPDG::DynamicBlockVector<T>>
  {
    typedef typename HPDG::DynamicBlockVector<T>::field_type field_type;
    typedef typename HPDG::DynamicBlockVector<T>::real_type real_type;
  };

}
#endif//DUNE_HPDG_DYNAMIC_BLOCKVECTOR_HH
