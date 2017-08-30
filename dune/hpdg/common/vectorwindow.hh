// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_VECTOR_WINDOW_HH_
#define DUNE_HPDG_VECTOR_WINDOW_HH_

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <complex>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <memory>

#include <dune/common/boundschecking.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/genericiterator.hh>

#include <vector>
#include <dune/common/densevector.hh>

namespace Dune {

  namespace HPDG {
  template<class K> class VectorWindow;
  }
  template<class K>
  struct DenseMatVecTraits< HPDG::VectorWindow<K> >
  {
    typedef HPDG::VectorWindow< K> derived_type;
    typedef std::vector< K> container_type;
    typedef K value_type;
    typedef typename container_type::size_type size_type;
  };

  template<class K>
  struct FieldTraits<HPDG::VectorWindow< K> >
  {
    typedef typename FieldTraits< K >::field_type field_type;
    typedef typename FieldTraits< K >::real_type real_type;
  };

  namespace HPDG {
  template<class K>
  class VectorWindow : public DenseVector< VectorWindow<K> >
  {
    //std::vector< K, Allocator > _data;
    K* data_;
    size_t n_;

    // Some algorithms require to make a copy from this vector. Though I don't 
    // very much like this fact, I don't want to rewrite every single of them.
    // Hence, if requested, this vector window can actually manage its own memory.
    std::unique_ptr<K[]> ownData_ = nullptr;

    typedef DenseVector< VectorWindow<K> > Base;
  public:
    typedef typename Base::size_type size_type;
    typedef typename Base::value_type value_type;

    //typedef Allocator allocator_type;

    /** Construct as a copy. This is, however, not the intended use of this class! */
    VectorWindow(const VectorWindow& x) {
      ownData_ = std::make_unique<K[]>(x.n_);
      data_=ownData_.get();
      n_=x.n_;
      for (size_t i = 0; i < n_; i++)
        data_[i] = x.data_[i];
    }

    /** Resize window. This implies new memory allocation. Do this at your own risk.
     * Do not cry at me if you break your performance because your memory is no longer
     * contiguous.
     */
    void resize(size_t i) {
      ownData_ = std::make_unique<K[]>(i);
      data_=ownData_.get();
      n_=i;
    }

    // TODO: I cannot imagine a situation where this should be used, but feel free to implement.
    VectorWindow(VectorWindow&& x) {
      this->n_ = x.n_;
      this->data_ = x.data_;
      x.data_ = nullptr;
    }

    VectorWindow() :
      data_(nullptr),
      n_(0) {}

    //! Constructor taking a ptr and the size of the vector
    VectorWindow(K* start, size_t n) :
      data_(start),
      n_(n) {}

    using Base::operator=;

    //! Copy assignment operator
    VectorWindow &operator=(const VectorWindow &other)
    {
      data_= other.data_;
      n_ = other.n_;
      return *this;
    }

    //! Move assignment operator
    VectorWindow &operator=(VectorWindow &&other)
    {
      data_=other.data_;
      n_ = other.n_;
      other.data_=nullptr;
      return *this;
    }

    void set(K* data, size_t n) {
      data_=data;
      n_=n;
    }

    value_type operator*(const VectorWindow& y) const {
      DUNE_ASSERT_BOUNDS(y.N() == n_);
      auto sum = value_type(0);
      const K* pthis = data_;
      const K* py = y.data_;
      // compute scalar product directly on the C-array
      for (size_t i = 0; i < n_; i++)
        sum+= *pthis++ * *py++; // aka. sum+= (*this)[i]*y[i]
      return sum;
    }

    size_type size() const {
      return n_;
    }

    K & operator[](size_type i) {
      DUNE_ASSERT_BOUNDS(i < size());
      return data_[i];
    }
    const K & operator[](size_type i) const {
      DUNE_ASSERT_BOUNDS(i < size());
      return data_[i];
    }
  };

} // end namespace
} // end namespace

#endif
