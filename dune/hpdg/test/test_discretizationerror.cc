#include <config.h>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/yaspgrid.hh>

#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/common/resizehelper.hh>
#include <dune/hpdg/dunefunctionsdiscretizationerror.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

using namespace Dune;

TestSuite
test_dunefunctionsdiscretizationerror()
{

  TestSuite suite;
  using Grid = YaspGrid<2>;
  auto grid = StructuredGridFactory<Grid>::createCubeGrid(
    { 0, 0 }, { 1.0, 1.0 }, { { 2, 2 } });

  auto basis = Functions::DynamicDGQkGLBlockBasis<Grid::LeafGridView>(
    grid->leafGridView());

  // make a DG function that is 1 on the first element and 0 on all others:
  using V = HPDG::DynamicBlockVector<FieldVector<double, 1>>;
  auto v = V();
  HPDG::resizeFromBasis(v, basis);
  v = 0.;

  // lazy man's version to get only one element:
  for (const auto& e : elements(basis.gridView())) {
    auto lv = basis.localView();
    lv.bind(e);
    v[lv.index(0)[0]] = 1.;
    break;
  }

  auto v_function =
    Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 1> >(basis, v);

  auto key = QuadratureRuleKey(2,16); // actually, we dont need a high order quadrature here


  // now, we compute the difference to a very complicated function. Namely, the constant 0 function.
  auto zero = [](const auto&) { return 0.0; };

  // The L^2 error should be exactly the root of area of the one element. In this easy case, we know that it is sqrt(1/4).
  auto l2Error = Fufem::DuneFunctionsDiscretizationError<Grid::LeafGridView>::computeL2Error(v_function, zero, key);
  suite.check(std::abs(l2Error-0.5)<1e-14);

  // TODO: set up test for H1 half norm. This would be trivially 0 in this case here.

  // Check the edge terms. Assuming Dirichlet 0 boundary, we expect the that on every edge of the offending element,
  // the jump of size 1 is integrated, given us sqrt(4*edge_length)= sqrt(2).
  auto edgekey = QuadratureRuleKey(1, 16); // again, complete overkill
  auto jumps = Fufem::DuneFunctionsDiscretizationError<Grid::LeafGridView>::computeJumpTerm(v_function, zero, edgekey);
  suite.check(std::abs(jumps - std::sqrt(2))<1e-14) << "Expected " << std::sqrt(2) << ", got " << jumps;

  return suite;
}

int
main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  TestSuite suite;

  suite.subTest(test_dunefunctionsdiscretizationerror());

  return suite.exit();
}