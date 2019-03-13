#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/assemblers/lumpedmass.hh>

#include <dune/solvers/norms/energynorm.hh>

#include "testobjects.hh"

using namespace Dune;

/** Test if the fast lumped quadrature for Gauss-Lobatto DG bases of given order is correct
 * by comparing the entries with the row sums of the mass matrix */
template<typename Grid>
TestSuite testLumpedMass(const Grid& grid, int order) {
  TestSuite suite;

  using Basis = Dune::Functions::DynamicDGQkGLBlockBasis<typename Grid::LeafGridView>;
  Basis basis(grid.leafGridView(), order);

  // compute the vector containing the weights of the lumped mass matrix
  auto l_mass = HPDG::gaussLobattoLumpedMass(basis);

  // compare with mass*1 (which should give the same result)
  auto mass = dynamicMassMatrix(grid, order);
  using Vector = HPDG::DynamicBlockVector<FieldVector<double,1>>;
  Vector w;
  HPDG::resizeFromBasis(w, basis);
  w=1.;
  {
    auto dummy=w;
    mass.mv(dummy,w);
  }

  // w should now be the same ass l_mass
  w-=l_mass;

  // we can even check in the actual L^2 norm, since we assembled the mass anyway
  auto norm = Dune::Solvers::EnergyNorm<decltype(mass), Vector>(mass);
  auto error = norm(w);
  suite.check(error < 1e-14) << "Error is " << error;

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  using Grid = YaspGrid<2>;
  auto grid = StructuredGridFactory<Grid>::createCubeGrid({0,0}, {1.0,1.0}, {{8,8}});


  TestSuite suite;
  for (int i = 1; i < 5; i++)
    suite.subTest(testLumpedMass(*grid, i));

  return suite.exit();
}
