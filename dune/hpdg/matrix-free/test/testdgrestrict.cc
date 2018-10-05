#include <config.h>
#include <cmath>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#if HAVE_UG
#include <dune/grid/uggrid.hh>
#else
#include <dune/grid/yaspgrid.hh>
#endif

#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/istl/bcrsmatrix.hh>

#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/dgrestrict.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/resizehelper.hh>
#include <dune/hpdg/assemblers/dgtocgtransferassembler.hh>
#include <dune/hpdg/transferoperators/dynamicblocktransfer.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>

using namespace Dune;

template<class GV>
TestSuite test_restrict(const GV& gv) {
  TestSuite suite;

  using Vector = HPDG::DynamicBlockVector<FieldVector<double, 1>>;

  using DGBasis = Functions::DynamicDGQkGLBlockBasis<GV>;
  DGBasis dgbasis(gv, 1);

  using Q1Basis = Functions::LagrangeBasis<GV, 1>;
  Q1Basis q1basis(gv);


  // interpolate a function:
  auto f = [](auto&& x) {
    return std::exp(x*x);
  };

  Vector x;
  HPDG::resizeFromBasis(x, dgbasis);

  Dune::Functions::interpolate(dgbasis, x, f);

  Vector x_restricted(q1basis.size(), 1);
  x_restricted.update();

  auto matrixFreeRestrict = Fufem::MatrixFree::DGRestrictionOperator<Vector, GV, Q1Basis, DGBasis>(q1basis, dgbasis);
  auto op = Fufem::MatrixFree::Operator<Vector, GV, decltype(matrixFreeRestrict)>(gv, matrixFreeRestrict);

  op.apply(x, x_restricted); // restrict to coarse mesh

  // Copy into a Dune::BlockVector so we can use it with bcrs
  auto dynBV_to_BV = [](const auto& dynamic) {
    using BV = BlockVector<FieldVector<double, 1>>;
    BV coarse(dynamic.size());

    for(std::size_t i = 0; i < dynamic.size(); i++) {
      coarse[i]=dynamic[i][0];
    }
    return coarse;
  };

  // Copy BlockVector into DynamicBlockVector
  auto x_restricted_bv = dynBV_to_BV(x_restricted);

  using BCRS = BCRSMatrix<FieldMatrix<double, 4, 1>>;
  BCRS transfer;
  Dune::HPDG::assembleDGtoCGTransferOperator(transfer, q1basis, dgbasis);

  // first, test if the restriction computed the same result:
  //
  // restrict via matrix:
  transfer.mtv(x, x_restricted_bv);

  {
    // test if we computed (roughly) the same:
    auto test = dynBV_to_BV(x_restricted);
    std::cout << "Norm matrix-free: " << test.two_norm() << std::endl;
    std::cout << "Norm classic: " << x_restricted_bv.two_norm() << std::endl;
    test-=x_restricted_bv;
    suite.check(test.two_norm() < 1e-14, "Check if matrixfree- and matrix-based compute the same restricted function");
  }

  // now, we prolong the vector and check again:

  auto x_prolonged = Vector::uninitializedCopy(x);

  auto coarseFunction = Functions::makeDiscreteGlobalBasisFunction<double>(q1basis, x_restricted);

  Dune::Functions::interpolate(dgbasis, x_prolonged, coarseFunction); // interpolation that does not use a matrix

  auto x_prolonged_matrix = Vector::uninitializedCopy(x);

  transfer.mv(x_restricted_bv, x_prolonged_matrix);

  {
    // test if we computed (roughly) the same:
    auto& test = x_prolonged;
    std::cout << "Norm matrix-free: " << test.two_norm() << std::endl;
    std::cout << "Norm classic: " << x_prolonged_matrix.two_norm() << std::endl;
    test-=x_prolonged_matrix;
    suite.check(test.two_norm() < 1e-14, "Check if matrixfree- and matrix-based compute the same prolonged function");
  }

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
#if HAVE_UG
  using Grid = UGGrid<dim>;
#else
  using Grid = YaspGrid<dim>;
#endif

  auto gridptr = Dune::StructuredGridFactory<Grid>::createCubeGrid({{0.0, 0.}}, {{1.,1.}}, {{16,16}});

#if HAVE_UG
  // unconformingly refine
  gridptr->setClosureType(Grid::ClosureType::NONE);
  gridptr->mark(1, *(gridptr->leafGridView().begin<0>()));
  gridptr->adapt();
  gridptr->postAdapt();
#endif

  TestSuite suite;
  suite.subTest(test_restrict(gridptr->leafGridView()));

  return suite.exit();
}
