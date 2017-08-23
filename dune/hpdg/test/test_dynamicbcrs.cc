#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/istl/matrixindexset.hh>
#include "utilities.hh"
#include <memory>

using namespace Dune;


TestSuite test_dynamicbcrs() {
  TestSuite suite("test_dynamicbcrs");

  // lets see, if we can create a bcrs matrix
  using Matrix = Dune::HPDG::DynamicBCRSMatrix<double>;
  Matrix dynbcrs{};
  auto& bcrs = dynbcrs.matrix(); // get the actual matrix

  Dune::MatrixIndexSet idx(2,2);
  idx.add(0,0);
  idx.add(1,0);
  idx.add(1,1);
  idx.exportIdx(bcrs);

  // additional setup
  dynbcrs.finishIdx();
  dynbcrs.blockRows(0) = 5; // first block row has 5 rows
  dynbcrs.blockRows(1) = 4; // second block row has only 4 rows
  dynbcrs.setSquare();
  dynbcrs.update(); // crucial!

  // bcrs is bound to the dynbcrs object, so we do not have to change anything on it.
  // Just make double sure you do not delete the dynbcrs object while still using the bcrs object.
  // dynbcrs will take all data with it and you're ending up with dangling pointers!
  //
  // set some values into the bcrs matrix just for fun:
  bcrs[0][0]=1.0;
  bcrs[1][0]=-1.5;
  bcrs[1][1]=2.0;

  // test with dynamic block vector
  auto bv = Dune::HPDG::makeDynamicBlockVector(dynbcrs); // this creates a dynamic block vector of the same dimensions as our matrix has
  {
    auto dummy = bv;
    dummy = 2.;
    bcrs.mv(dummy, bv);
  }

  // output vector: (yes, we do have iterators :) )
  for (const auto& e: bv) {
    for (const auto& ee: e) std::cout << ee << std::endl;
  }


  // make some checks
  {
    auto vec1 = bv;
    auto vec2 = vec1;
    vec1=1;
    vec2=2;
    suite.check(std::abs(vec1.dimension()*1.*2. - (vec1*vec2))<1e-15, "Check scalar product");
    auto vec3 = vec1+vec2;
    vec3-=vec2;
    vec3-=vec1;
    suite.check((vec3*vec3) < 1e-15, "Check addition and subtraction") << "(v1+v2) - v1 - v2  has norm " << (vec3*vec3) << std::endl;
  }

  // test copy construction
  {
    auto matrix2 = dynbcrs;
    // compare last blocks
    const auto& b1 = bcrs[1][1];
    const auto& b2 = matrix2.matrix()[1][1];
    suite.check(b1.N() == b2.N());
    suite.check(b1.M() == b2.M());

    for (size_t i = 0; i < b1.N(); i++) {
      for (size_t j = 0; j < b1.M(); j++) {
        suite.check(b1[i][j]==b2[i][j]);
      }
    }
  }
  // Copying the Dune::BCRSMatrix object directly _must not_ work, as the block types are windows and do not manage storage themselves.
  // The way to go is to copy the DynamicBCRS object. Hence, the following must throw:
  suite.check(

    doesThrow([&]() { auto illegalCopy = bcrs; }

  ));

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_dynamicbcrs());
  return suite.exit();
}
