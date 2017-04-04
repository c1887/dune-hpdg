#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/bvector.hh>
#include <dune/common/fmatrix.hh>

#include <dune/hpdg/common/blockwiseoperations.hh>
using namespace Dune;

/** return a FieldMatrix
 * | 2 1 |
 * | 4 2 |
 */
auto ExampleMatrix() {
  using M = FieldMatrix<double, 2, 2>;
  M m({{2,1},{4,2}});
  return m;
}

TestSuite test_scaleByTransposedBlock() {
  TestSuite suite("test_scaleByTransposedBlock");
  auto matrix = ExampleMatrix();
  auto b = BlockVector<FieldVector<double, 2> >({{-2,1}, {1,-2}});

  // The components of b should be scaled by matrix. Hence, the result shall have the same
  // dimensions as b
  auto x = HPDG::scaleByTransposedBlock(matrix, b);
  suite.check(x.size() == b.size(), "Check dimensions");

  // The first component of b is in matrix^T's kernel, but not matrix's kernel
  suite.check(x[0].infinity_norm() < 1e-15 and x[1].infinity_norm()>5, "Check if transposed is used") 
    << "b[0] not in kernel of M^T";

  // Check if the method with void return gives the same results
  auto y=b;
  HPDG::scaleByTransposedBlock(matrix, b, y);
  y-=x;
  suite.check(y.two_norm() < 1e-15, "Check if return-by-value and reference give same result") 
    << "Results do not match (Error: " << y.two_norm() <<")";
  
  return suite;
}

TestSuite test_scaleByBlock() {
  TestSuite suite("test_scaleByTransposedBlock");
  auto matrix = ExampleMatrix();
  auto b = BlockVector<FieldVector<double, 2> >({{-2,1}, {1,-2}});

  // The components of the block-vector b should be scaled by matrix^T. Hence, the result shall have
  // the same dimensions as b
  auto x = HPDG::scaleByBlock(matrix, b);
  suite.check(x.size() == b.size(), "Check dimensions");

  // The second component of b is in matrix's kernel, but not matrix^T's kernel
  suite.check(x[1].infinity_norm() < 1e-15 and x[0].infinity_norm()>5, "Check if untransposed matrix is used") 
    << "b[1] not in kernel of M^T";

  // Check if the method with void return gives the same results
  auto y=b;
  HPDG::scaleByBlock(matrix, b, y);
  y-=x;
  suite.check(y.two_norm() < 1e-15, "Check if return-by-value and reference give same result") 
    << "Results do not match (Error: " << y.two_norm() <<")";
  
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_scaleByBlock());
  suite.subTest(test_scaleByTransposedBlock());
  return suite.exit();
}
