#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/functionspacebases/persistentgridviewdatatransfer.hh>
#include <dune/grid/yaspgrid.hh>
using namespace Dune;

TestSuite test_persistentGVDataTransfer() {
  TestSuite suite("test_persistentGVDataTransfer");

  using Grid = Dune::YaspGrid<2>;
  Grid grid({1.,1.}, {{2,2}});

  auto pgv = Dune::Functions::Experimental::persistentGridView<Grid::LeafGridView>(grid.leafGridView());

  auto pgv_data = Dune::HPDG::PersistentGridViewDataTransfer<Grid::LeafGridView, double>(pgv);


  for(const auto& element : elements(grid.leafGridView())) {
    pgv_data[element] = 0.42;
  }

  grid.globalRefine(1);


  for(const auto& element : elements(grid.leafGridView())) {
    suite.check(pgv_data[element] == 0.42, "Check if data could be read from parent") << "Expected 0.42, got " << pgv_data[element];
  }

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_persistentGVDataTransfer());
  return suite.exit();
}
