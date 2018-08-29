#pragma once
#include <iostream>
#include <memory>
#include <dune/common/exceptions.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#if HAVE_BLAS
extern "C" {
extern void dgemm_(char * transa, char * transb, int * m, int * n, int * k,
                  double * alpha, const double * A, int * lda,
                                const double * B, int * ldb, double * beta,
                                              double *, int * ldc);
}
#endif
namespace Dune{
namespace HPDG{

/** Dense matrix which only real functionality is to
 * offer fast matrix-matrix products via BLAS.
 *
 * Entry access is via matrix(i,j).
 * There are no iterators
 */
template<class T=double>
class MMMatrix {
  size_t n_;
  size_t m_;
  std::unique_ptr<T[]> data_;

  public:

  MMMatrix(size_t n, size_t m) :
    n_(n),
    m_(m),
    data_(new double[n*m])
  {}

  const T& operator()(size_t i, size_t j) const {
    return data_[j*n_ + i];
  }
  T& operator()(size_t i, size_t j) {
    return data_[j*n_ + i];
  }

  MMMatrix<T>& operator=(T scalar) {
    for(size_t i = 0; i< n_*m_; i++)
      data_[i]=scalar;
    return *this;
  }

  MMMatrix(const MMMatrix& other) {
    n_=other.n_;
    m_=other.m_;
    data_= std::make_unique<T[]>(m_*n_);
    for(size_t i = 0; i< n_*m_; i++)
      data_[i]=other.data_[i];
  }

  MMMatrix& operator=(const MMMatrix& other) {
    n_=other.n_;
    m_=other.m_;
    if (data_ != nullptr)
      data_.release();
    data_= std::make_unique<T[]>(m_*n_);
    for(size_t i = 0; i< n_*m_; i++)
      data_[i]=other.data_[i];

    return *this;
  }

  MMMatrix() :
    n_(0),
    m_(0),
    data_(nullptr)
  {}

  MMMatrix(MMMatrix&& rv) :
    n_(rv.n_),
    m_(rv.m_),
    data_(std::move(rv.data_))
  {}

  auto N() const {
    return n_;
  }

  auto M() const {
    return m_;
  }

  T* data() {
    return data_.get();
  }

  const T* data() const {
    return data_.get();
  }

  friend MMMatrix<T> operator*(const MMMatrix<T>& m0, const MMMatrix<T>& m1) {
    MMMatrix m(m0.n_, m1.m_);
#if HAVE_BLAS
    m=0.;
    char Nchar = 'N';
    int M = m0.n_;
    int N = m1.m_;
    int K = m0.m_;
    double alpha=1.0;
    double beta=0.0;
    int lda = M;
    int ldb = K;
    dgemm_(&Nchar, &Nchar, &M, &N, &K, &alpha, m0.data_.get(), &lda, m1.data_.get(), &ldb, &beta, m.data_.get(),&M);
#else
    // fallback impl.
    // TODO
    DUNE_THROW(Dune::NotImplemented, "Fallback to BLAS dgemm_ not yet there");
#endif

    return m;
  }
};

template<class T>
auto transposedBlasMatrixProduct (const Dune::Matrix<T>& m0, const Dune::Matrix<T>& m1) {
  Dune::Matrix<T> ret(m0.N(), m1.M());
  ret=0.;

  assert(sizeof(double) == sizeof(std::decay_t<decltype(m0[0][0])>));

#if HAVE_BLAS
  char Nchar = 'T';
  int M = m0.N();
  int N = m1.M();
  int K = m0.M();
  double alpha=1.0;
  double beta=0.0;
  int lda = M;
  int ldb = K;
  dgemm_(&Nchar, &Nchar, &M, &N, &K, &alpha, (double*) &(m0[0][0]), &lda, (double*) &(m1[0][0]), &ldb, &beta,(double*) &(ret[0][0]),&M);
#else
  DUNE_THROW(Dune::NotImplemented, "No BLAS fallback :(");
#endif
  return ret;
}


// compute B^T U L, where B, U and L are expected to be row-mayor (as common in Dune)
// Moreover, it is assumed that
// B, L \in \R^{n x m}
// and U in \R{n x n}.
// Hence, this is not completely general. Be careful.
// TODO: The function name is misleading. What you actually get is (B^T U L)^T, see the non blas variant:
template<class BM, class UM,class LM>
auto BtUL (const BM& b, const UM* u, const LM& l) {

  // ALWAYS REMEMBER: What looks like l here, is in fact l^T for BLAS!

  assert(sizeof(double) == sizeof(UM));
  assert(sizeof(double) == sizeof(typename BM::block_type));

  // sizes:
  // L = p x quads
  // L^T = quads x p = l
  // U = p * p
  // U^T = p * p = u
  // B^T = quads * p = b
  // L^T U^T = quads * p = (UL)^T



#if HAVE_BLAS
  Dune::Matrix<Dune::FieldMatrix<double, 1, 1>> ret(l.M(), b.M());
  ret=0.;

  auto ltut = std::vector<double>(l.M()*l.N(), 0.); // here we compute (UL)^T = L^T U^T

  char Tchar = 'T';
  char Nchar = 'N';
  int p = l.N();
  int quads = l.M();
  double alpha=1.0;
  double beta=0.0;

  // first step, compute L^T U^T = (UL)^T
  dgemm_(&Nchar, &Nchar, &quads, &p, &p, &alpha, (double*) &(l[0][0]), &quads, static_cast<const double*>(u), &p, &beta, ltut.data(),&quads);

  // second  step, compute B^T (UL) = B^T * ((UL)^T)^T
  // BUT: (UL)^T has been stored column-mayor, so be careful
  dgemm_(&Nchar, &Tchar, &quads, &quads, &p, &alpha, (double*) &(b[0][0]), &quads, ltut.data() , &quads, &beta,(double*) &(ret[0][0]),&quads);
#else
  // copy u into FM format
  BM U(b.N(), b.N());
  {
    auto* ptr = &(U[0][0]);
    for (size_t i= 0; i < b.N()*b.N(); i++)
      ptr[i]=u[i];
  }
  auto BtU = b.transpose()*U;
  auto ret = (BtU*l).transpose();
#endif
  return ret;
}


// Compute A^T * B^T
template<class T>
auto AtBt (const Dune::Matrix<T>& m0, const Dune::Matrix<T>& m1) {
  Dune::Matrix<T> ret(m1.N(), m0.M());
  ret=0.;

  static_assert(sizeof(double) == sizeof(T), "double?");

  const double* A = static_cast<const double*>(&(m0[0][0][0][0]));
  const double* B = static_cast<const double*>(&(m1[0][0][0][0]));

#if HAVE_BLAS
  char Nchar = 'N';
  int K = m0.N();
  int N = m1.N();
  int M = m0.M();
  double alpha=1.0;
  double beta=0.0;
  int lda = M;
  int ldb = K;
  dgemm_(&Nchar, &Nchar, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta,static_cast<double*>(&(ret[0][0][0][0])),&M);
#else
  DUNE_THROW(Dune::NotImplemented, "No BLAS fallback :(");
#endif
  return ret;
}

// C+= A X^T B^T
// The memory at C is expected to be allocated and ready to be used.
template<class T, class CType>
void CplusAXtBt (const Dune::Matrix<T>& op_A, const Matrix<T>& op_X, const Matrix<T>& op_B, CType* C) {
  static_assert(sizeof(double) == sizeof(T), "double?");

  const double* At = static_cast<const double*>(&(op_A[0][0][0][0]));
  const double* Bt = static_cast<const double*>(&(op_B[0][0][0][0]));
  const double* Xt = static_cast<const double*>(&(op_X[0][0][0][0]));

  // Again, remember the data is coming in row-mayor, but BLAS uses col-mayor.
  // That means, instead of C=A X^T B^T, we'll compute
  // C^T=BXA^T.

  // expected sizes:
  // C: p x p
  // A: p x quads
  // B: p x quads
  // X: quads x quads

#if HAVE_BLAS
  char Tchar = 'T';
  char Nchar = 'N';
  int p = op_A.N();
  int quads = op_A.M();
  double alpha=1.0;
  double beta=1.0;

  auto bx = std::vector<double>(quads*p, 0.); // here we compute BX

  // compute bx
  dgemm_(&Tchar, &Tchar, &p, &quads, &quads, &alpha, Bt, &quads, Xt, &quads, &beta, bx.data(), &p);

  // now that we have bx, compute
  // bx * A^T
  dgemm_(&Nchar, &Nchar, &p, &p, &quads, &alpha, bx.data(), &p, At, &quads, &beta, C, &p);

#else
  //std::cout << "WARNING: Using BLAS fallback." << std::endl;
  auto AXt = op_A*op_X.transpose();
  auto AXtBt = AXt*op_B.transpose();

  const auto* ptr = &(AXtBt[0][0][0][0]);
  for (size_t i = 0; i < AXtBt.N()*AXtBt.M(); i++)
    C[i]+=ptr[i];

#endif
}

}}
