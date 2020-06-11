#pragma once
#include <cstddef>
#include <dune/common/fmatrix.hh>

#include <dune/matrix-vector/algorithm.hh>
#include <dune/matrix-vector/axpy.hh>

namespace Dune {
namespace HPDG {
namespace Arithmetic {

/** Computes out += factor*matrix*vector.
 *
 * It is assumed that out has already the right size!
 *
 * This works exactly for the case that we have the following index blocking:
 * v[first_idx][second_idx][last_idx]
 *
 * The last idx should index a field{vector, matrix} of any size.
 * However, the matrix might also have a 1x1 FieldMatrix there and is treated as
 * a scalar.
 */
template<typename M, typename V>
void
matrixVectorProduct(const M& matrix,
                    const V& vector,
                    V& out,
                    double factor = 1.)
{
  for (std::size_t i = 0; i < matrix.N(); ++i) {
    Dune::MatrixVector::sparseRangeFor(matrix[i], [&](auto&& aij, auto j) {
      // now, we're in the matrix window.
      for (std::size_t ii = 0; ii < aij.N(); ++ii) {
        for (std::size_t jj = 0; jj < aij.M(); ++jj) {
          MatrixVector::addProduct(
            out[i][ii], factor, aij[ii][jj], vector[j][jj]);
        }
      }
    });
  }
}

template<int n>
void
fieldMatrix_umtv(const FieldMatrix<double, n, n>& matrix,
                 const FieldVector<double, n>& vector,
                 FieldVector<double, n>& out,
                 double factor = 1.)
{
  matrix.usmtv(factor, vector, out);
}

template<int n, typename = std::enable_if_t<n != 1>>
void
fieldMatrix_umtv(const FieldMatrix<double, 1, 1>& matrix,
                 const FieldVector<double, n>& vector,
                 FieldVector<double, n>& out,
                 double factor = 1.)
{
  out.axpy(factor * matrix[0][0], vector);
}

/** Computes out += factor*matrix^T*vector. */
template<typename M, typename V>
void
transposedMatrixVectorProduct(const M& matrix,
                              const V& vector,
                              V& out,
                              double factor = 1.)
{
  for (std::size_t i = 0; i < matrix.N(); ++i) {
    Dune::MatrixVector::sparseRangeFor(matrix[i], [&](auto&& aij, auto j) {
      // now, we're in the matrix window.
      for (std::size_t ii = 0; ii < aij.N(); ++ii) {
        for (std::size_t jj = 0; jj < aij.M(); ++jj) {
          fieldMatrix_umtv(aij[ii][jj], vector[i][ii], out[j][jj], factor);
        }
      }
    });
  }
}


/** Compute A += T1^T * B * T2
 *
 * T1, T2 may have scalar entries
 * (i.e. FieldMatrix of size 1x1),
 * while A and B can have also  be matrix-valued
 * (i.e. FieldMatrix of size nxn).
 */
template<class K, class K2>
void
addTransformedMatrix(MatrixWindow<K>& A,
                     const MatrixWindow<K2>& T1,
                     const MatrixWindow<K>& B,
                     const MatrixWindow<K2>& T2)
{

  auto T1transposedB = Dune::Matrix<K>(T1.M(), B.M());

  T1transposedB = 0;
  for (size_t i = 0; i < T1.M(); ++i)
    for (size_t k = 0; k < B.N(); ++k)
      if (T1[k][i] != 0) {
        for (size_t l = 0; l < B.M(); ++l)
          Dune::MatrixVector::addProduct(
            T1transposedB[i][l], T1[k][i], B[k][l]);
      }

  // multiply such that row-major format is better used
  for (size_t i = 0; i < A.N(); i++) {
    for (size_t j = 0; j < A.M(); j++) {
      auto& Aij = A[i][j];
      for (size_t l = 0; l < T2.N(); l++)
        Dune::MatrixVector::addProduct(Aij, T2[l][j], T1transposedB[i][l]);
    }
  }
}
}
}
}
