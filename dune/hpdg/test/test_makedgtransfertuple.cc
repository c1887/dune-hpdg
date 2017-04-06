#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/indices.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/unused.hh>

#include <dune/hpdg/transferoperators/makedgtransfertuple.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include "utilities.hh"

using namespace Dune;

TestSuite test_makedgtransfertuple() {
  TestSuite suite;

  using Grid = YaspGrid<2>;
  auto grid = StructuredGridFactory<Grid>::createCubeGrid({0,0}, {1.0,1.0}, {{4,4}});

  using namespace Dune::Indices;
  {
    // set up some level pairs
    constexpr auto level_pairs = std::make_tuple(
        std::make_pair(_2, _1),
        std::make_pair(_4, _2)
        );

    // Compute transfer operators
    auto transfer = HPDG::make_dgTransferTuple(level_pairs, *grid);
    // Check if the tuple sizes coincide
    suite.check(Hybrid::size(transfer) == Hybrid::size(level_pairs), "Check if sizes match") << "Transfer tuple has " <<
    ((Hybrid::size(transfer) > Hybrid::size(level_pairs)) ? "more" : "less") << " elements than there are level pairs" << std::endl;
  }

  // try if setup with bad level pairs throws
  {
    // case 1: pair is not ordered correctly
    constexpr auto level_pairs = std::make_tuple(
        std::make_pair(_1, _2), // pair is ordered in the wrong way
        std::make_pair(_4, _2)
        );

    // try if computing transfer operators throws
    auto toEvaluate = [&](auto&& a, auto&& b) {HPDG::make_dgTransferTuple(a,b);};
    suite.check(doesThrow(toEvaluate, level_pairs, *grid), 
        "Check handling of wrongly ordered pairs") << "Did not throw where it should have!";

  }
  {
    // case 2: coarse and fine values do not match
      constexpr auto level_pairs = std::make_tuple(
          std::make_pair(_2, _1),
          std::make_pair(_4, _3) // coarse level has to be _2
          );

      // try if computing transfer operators throws
      auto toEvaluate = [&](auto&& a, auto&&b) {HPDG::make_dgTransferTuple(a, b);};
      suite.check(doesThrow(toEvaluate, level_pairs, *grid), 
          "Check handling of consecutive pairs where fine and coarse levels do not match") << "Did not throw where it should have!";
  }
  
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_makedgtransfertuple());
  return suite.exit();
}
