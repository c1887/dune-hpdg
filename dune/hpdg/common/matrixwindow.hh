// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_MATRIX_WINDOW_HH
#define DUNE_HPDG_MATRIX_WINDOW_HH

#include <cmath>
#include <cstddef>
#include <iostream>
#include <initializer_list>

#include <dune/common/boundschecking.hh>
#include <dune/common/exceptions.hh>
//#include <dune/common/dynvector.hh>
#include <dune/common/densematrix.hh>
#include <dune/common/typetraits.hh>

#include <dune/hpdg/common/vectorwindow.hh>
namespace Dune
{

  // forward declartion
  namespace HPDG {
    template< class K > class MatrixWindow;
  }

  // note: this cannot be in HPDG namespace :(
  //
  template< class K >
  struct DenseMatVecTraits< HPDG::MatrixWindow<K> >
  {
    typedef HPDG::MatrixWindow<K> derived_type;

    typedef HPDG::VectorWindow<K> row_type;

    typedef row_type &row_reference;
    typedef const row_type &const_row_reference;

    typedef std::vector<K> container_type;
    typedef K value_type;
    typedef typename container_type::size_type size_type;
  };

  template<class K>
  struct FieldTraits< HPDG::MatrixWindow<K> >
  {
    typedef typename FieldTraits<K>::field_type field_type;
    typedef typename FieldTraits<K>::real_type real_type;
  };

  namespace HPDG {

    /** Given a array of values and matrix dimensions, this gives the semantics
     * of a Dune::DenseMatrix. However, memory management is totally up to the user.
     * This is barely a _window_, it does not own anything!
     */
    template<class K>
    class MatrixWindow : public DenseMatrix< MatrixWindow<K> >
    {
      typedef DenseMatrix< MatrixWindow<K> > Base;
      public:
      typedef typename Base::size_type size_type;
      typedef typename Base::value_type value_type;
      typedef typename Base::row_type row_type;

      //===== constructors
      //! \brief Default constructor
      MatrixWindow () :
      data_(nullptr),
      n_(0),
      m_(0),
      rowAccess(nullptr, 0) {}

      //! Constructor taking a ptr to data and the matrix dimensions
      MatrixWindow(K* data, size_t n, size_t m) :
        data_(data),
        n_(n),
        m_(m),
        rowAccess(nullptr, 0) {}


      void resize (size_type r, size_type c, value_type v = value_type() )
      {
        DUNE_THROW(Dune::Exception, "MatrixWindow cannot resize itself. Use set()!");
      }

      void set (K* data, size_type n, size_type m) {
        data_=data;
        n_ = n;
        m_ = m;
      }

      template <typename T, typename = std::enable_if_t<Dune::IsNumber<T>::value>>
      MatrixWindow& operator=(T scalar) {
        for (size_t i = 0; i < n_*m_; i++) data_[i]=scalar;
        return *this;
      }

      template <typename M, typename = std::enable_if_t<!Dune::IsNumber<M>::value>>
      MatrixWindow& operator=(const M& matrix) {
        DUNE_ASSERT_BOUNDS(n_==matrix.N() and m_ == matrix.M());
        for (size_t i = 0; i < n_; i++) {
          for (size_t j = 0; j < m_; j++) {
            data_[i*n_ + j]= matrix[i][j];
          }
        }
        return *this;
      }

      MatrixWindow& operator=(const MatrixWindow& other) {
        if (data_==nullptr)
          DUNE_THROW(Dune::Exception, "Trying to write through uninitialized MatrixWindow (you need to set the ptr to some valid place!)");

        DUNE_ASSERT_BOUNDS(other.n_ == n_ and other.m_ == m_)
        if (data_== other.data_) return *this;

        for (size_t i = 0; i < n_*m_; i++) data_[i]=other.data_[i];
        return *this;
      }

      size_type mat_rows() const {
        return n_;
      }
      size_type N() const {
        return this->mat_rows();
      }
      size_type mat_cols() const {
        return m_;
      }
      size_type M() const {
        return this->mat_cols();
      }

      const row_type& mat_access(size_type i) const {
        DUNE_ASSERT_BOUNDS(i < n_);
        rowAccess.set(data_ + i*n_, m_); // pointer arithmetic, f*ck yeah!
        return rowAccess;
      }

      row_type& mat_access(size_type i) {
        DUNE_ASSERT_BOUNDS(i < n_);
        rowAccess.set(data_ + i*n_, m_); // pointer arithmetic, f*ck yeah!
        return rowAccess;
      }

      /** y=Ax */
      void mv (const VectorWindow<K>& x, VectorWindow<K>& y) const
      {
        DUNE_ASSERT_BOUNDS((void*)(&x) != (void*)(&y));
        DUNE_ASSERT_BOUNDS(x.N() == M());
        DUNE_ASSERT_BOUNDS(y.N() == N());

        for (size_type i=0; i<n_; ++i)
        {
          y[i] = mat_access(i) * x;
        }
      }

      /** y+=Ax */
      void umv (const VectorWindow<K>& x, VectorWindow<K>& y) const
      {
        DUNE_ASSERT_BOUNDS((void*)(&x) != (void*)(&y));
        DUNE_ASSERT_BOUNDS(x.N() == M());
        DUNE_ASSERT_BOUNDS(y.N() == N());

        for (size_type i=0; i<n_; ++i)
        {
          y[i] += mat_access(i) * x;
        }
      }

      /** y-=Ax */
      void mmv (const VectorWindow<K>& x, VectorWindow<K>& y) const
      {
        DUNE_ASSERT_BOUNDS((void*)(&x) != (void*)(&y));
        DUNE_ASSERT_BOUNDS(x.N() == M());
        DUNE_ASSERT_BOUNDS(y.N() == N());

        for (size_type i=0; i<n_; ++i)
        {
          y[i] -= mat_access(i) * x;
        }
      }

      private:
      K* data_;
      size_t n_; // number of rows
      size_t m_; // number of columns
      mutable row_type rowAccess;

    };

  } // end namespace
} // end namespace

#endif
