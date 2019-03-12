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
  auto gridptr = Dune::StructuredGridFactory<Grid>::createCubeGrid({{0,0}}, {{1,1}}, {{4,4}});

  size_t iter = 1;

  constexpr const int k = 2;
  auto dynamicMatrix = dynamicStiffnessMatrix(*gridptr, k);
  auto staticMatrix = stiffnessMatrix<k>(*gridptr);

  using StaticVector = Dune::BlockVector<Dune::FieldVector<double, Dune::StaticPower<k+1,dim>::power>>;
  StaticVector staticB(staticMatrix.N());
  auto dynamicB = Dune::HPDG::makeDynamicBlockVector(dynamicMatrix);
  staticB=1.0;
  dynamicB=1.0;
  auto dynamicX= dynamicB;
  auto staticX = staticB;

  // perform a dynamic BlockGS iteration
  auto dynamicGS = Dune::HPDG::DynamicBlockGS<decltype(dynamicMatrix), decltype(dynamicX)>();
  dynamicGS.setProblem(dynamicMatrix, dynamicX, dynamicB);
  // TODO: ignore nodes
  for (size_t i = 0; i < iter; i++)
    dynamicGS.iterate();

  // perform a static BlockGS iteration
  using BitVector = Dune::Solvers::DefaultBitVector_t<StaticVector>;
  auto staticGS = Dune::Solvers::BlockGSStepFactory<decltype(staticMatrix), StaticVector, BitVector>::create(Dune::Solvers::BlockGS::LocalSolvers::gs(0.,0.), Dune::Solvers::BlockGS::Direction::FORWARD);
  staticGS.setProblem(staticMatrix, staticX, staticB);
  BitVector staticIgnore(staticX.size(), false);
  staticGS.setIgnore(staticIgnore);
  for (size_t i = 0; i < iter; i++)
    staticGS.iterate();

  // compare results (may be slightly off due to floating point arithmetic and slightly different implementations
  auto max = 0.;
  for (size_t i = 1; i <staticX.size()-1; i++) {
    for (size_t j = 0; j <staticX[i].size(); j++) {
      max = std::max(std::abs(staticX[i][j]-dynamicX[i][j]),max);
    }
  }
  suite.check(max < 1e-14, "Check difference between static and dynamic Block GS") << "Max norm of difference is " << max;

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_dynamicblockgs());
  return suite.exit();
}
