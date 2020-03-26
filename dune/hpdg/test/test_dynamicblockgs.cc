#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/timer.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/iterationsteps/dynamicblockgs.hh>
#include <dune/solvers/iterationsteps/blockgssteps.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include "testobjects.hh"
#include <memory>

using namespace Dune;


TestSuite test_dynamicblockgs() {
  TestSuite suite("test_dynamicblockgs");

  constexpr const int dim =2;
  using Grid = Dune::YaspGrid<dim>;
  auto gridptr = Dune::StructuredGridFactory<Grid>::createCubeGrid({{0,0}}, {{1,1}}, {{2,2}});

  size_t iter = 100;

  constexpr const int k = 2;
  auto dynamicMatrix = dynamicStiffnessMatrix(*gridptr, k);
  auto dynamicB = Dune::HPDG::makeDynamicBlockVector(dynamicMatrix);
  dynamicB=1.0;
  auto dynamicX= dynamicB;

  // perform a dynamic BlockGS iteration
  auto dynamicGS = Dune::HPDG::DynamicBlockGS<decltype(dynamicMatrix), decltype(dynamicX)>();
  dynamicGS.setProblem(dynamicMatrix, dynamicX, dynamicB);
  for (size_t i = 0; i < iter; i++)
    dynamicGS.iterate();

  // compute residual:
  dynamicMatrix.mmv(dynamicX, dynamicB);

  // as the system is really small, we should approximately have solved the system already.
  suite.check(dynamicB.two_norm() < 1e-13, "Test if small system was solved") << "Norm is " << dynamicB.two_norm();
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_dynamicblockgs());
  return suite.exit();
}
