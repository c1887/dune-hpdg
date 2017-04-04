#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/common/arraytotuple.hh>
using namespace Dune;

TestSuite test_arrayToTuple() {
  TestSuite suite("test_arrayToTuple");

  // create array and a tuple with the same values
  std::array<int, 3> arr{{1,2,3}};
  std::tuple<int, int, int> tup{1,2,3};

  // convert array to tuple
  auto newtup = HPDG::arrayToTuple(arr);
  suite.check(tup == newtup, "Compare tuple constructed from array with tuple") << "Tuples do no match!";
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_arrayToTuple());
  return suite.exit();
}
