#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/istl/matrixindexset.hh>
#include <memory>

using namespace Dune;


TestSuite test_dynamicbcrs() {
  TestSuite suite("test_dynamicbcrs");

  const size_t blockrows = 5;

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
  for (size_t i = 0; i < 2; i++)
    dynbcrs.blockRows(i) = blockrows;
  dynbcrs.update(); // crucial!

  // bcrs is bound to the dynbcrs object, so we do not have to change anything on it.
  // Just make double sure you do not delete the dynbcrs object while still using the bcrs object.
  // dynbcrs will take all data with it and you're ending up with dangling pointers!
  //
  // set some values into the bcrs matrix just for fun:
  bcrs[0][0]=1.0;
  bcrs[1][0]=-1.5;
  bcrs[1][1]=2.0;

  // show that this is actually usable:
  using BV = Dune::BlockVector<Dune::FieldVector<double, blockrows>>;
  BV bv(2);
  bv=2.;
  {
    auto dummy = bv;
    bcrs.mv(dummy, bv);
  }

  for (const auto& e: bv) std::cout << e << std::endl;
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_dynamicbcrs());
  return suite.exit();
}
