#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>


#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/sflaplace.hh>
#include <dune/hpdg/matrix-free/localoperators/sfipdg.hh>
#include <dune/hpdg/matrix-free/localoperators/uniformlaplaceoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/laplaceoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdgoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdgnorm.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/common/resizehelper.hh>
#include <dune/hpdg/test/randomvector.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
using namespace Dune;

template<class GV>
TestSuite test_IPDG(const GV& gv, int k) {
  TestSuite suite;
  using Vector = Dune::HPDG::DynamicBlockVector<FieldVector<double,1>>;

  auto basis = Dune::Functions::DynamicDGQkGLBlockBasis<GV>(gv, k);
  basis.preBasis().degree(*(gv.template begin<0>())) = k+1;

  std::cout << "\nTesting IPDG with Order " << k << " (" << basis.dimension() <<" unknowns):" << std::endl;
  Vector x;
  Dune::HPDG::resizeFromBasis(x, basis);

  Dune::HPDG::fillVectorRandomly(x); // fill vector with (fixed-seed) random values. This will in particular give a discontinuous function

  auto Ax=x;
  Ax=0.0;

  std::array<double, 2> time;

  auto sf_laplace = Dune::Fufem::MatrixFree::SumFactIPDGOperator<Vector, GV, decltype(basis)>(basis, 2.0, true);
  auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(sf_laplace)>(gv, sf_laplace);
  {
    Dune::Timer timer;
    op.apply(x, Ax);
    time[0] = timer.stop();
    std::cout << "Sum-factorized Laplace with order " << k <<" took: " << time[0] << std::endl;
  }

  // for test, do the same with naive IPDG matrix-free
  auto laplace = Dune::Fufem::MatrixFree::IPDGOperator<Vector, GV, decltype(basis)>(basis, 2.0, true);
  auto op_mf = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(laplace)>(gv, laplace);
  auto Ax_mf = Ax;
  Ax_mf=0.;
  {
    Dune::Timer timer;
    op_mf.apply(x, Ax_mf);
    time[1] = timer.stop();
    std::cout << "Naive matrix-free IPDG with order " << k <<" took: " << time[1] << std::endl;
  }
  std::cout << "Sumfactored Laplace was " << time[1]/time[0] << " times faster!" << std::endl;

  // Test with ipdg energy norm:
  double error;
  {
    Ax-=Ax_mf;
    auto ipdg = Dune::Fufem::MatrixFree::IPDGNorm<Vector, GV, decltype(basis)>(basis);
    auto ipdgop = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(ipdg)>(gv, ipdg);

    auto dummy= Ax;
    ipdgop.apply(Ax, dummy);
    error = Ax*dummy;
  }
  std::cout << error << std::endl;
  suite.check(error<1e-12, "Check if sumfacttorized and standard matrix free Laplace yield the same") << "Difference for order " <<k <<" is " << error << std::endl;

  return suite;
}

// Tests if sumfactorized and standard matrix-free bulk laplace yield the same vectors
template<class GV>
TestSuite test_bulk(const GV& gv, int k) {
  TestSuite suite;
  using Vector = Dune::HPDG::DynamicBlockVector<FieldVector<double,1>>;

  auto basis = Dune::Functions::DynamicDGQkGLBlockBasis<GV>(gv, k);
  basis.preBasis().degree(*(gv.template begin<0>())) = k+1;

  std::cout << "\nTesting Bulk Laplace with Order " << k << " (" << basis.dimension() <<" unknowns):" << std::endl;
  Vector x;
  Dune::HPDG::resizeFromBasis(x, basis);
  auto func=[](auto&& x) {
    return exp(x*x);
  };
  auto xbe = Dune::Functions::hierarchicVector(x);
  Dune::Functions::interpolate(basis, xbe, func);
  auto Ax=x;
  Ax=0.0;

  std::array<double, 2> time;

  auto sf_laplace = Dune::Fufem::MatrixFree::SumFactLaplaceOperator<Vector, GV, decltype(basis)>(basis);
  auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(sf_laplace)>(gv, sf_laplace);
  {
    Dune::Timer timer;
    op.apply(x, Ax);
    time[0] = timer.stop();
    std::cout << "Sum-factorized Laplace with order " << k <<" took: " << time[0] << std::endl;
  }

  // for test, do the same with standard Laplace matrix-free
  //auto laplace = Dune::Fufem::MatrixFree::UniformLaplaceOperator<Vector, GV, decltype(basis)>(basis);
  auto laplace = Dune::Fufem::MatrixFree::LaplaceOperator<Vector, GV, decltype(basis)>(basis);
  auto op_mf = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(laplace)>(gv, laplace);
  auto Ax_mf = Ax;
  Ax_mf=0.;
  {
    Dune::Timer timer;
    op_mf.apply(x, Ax_mf);
    time[1] = timer.stop();
    std::cout << "Standard Laplace with order " << k <<" took: " << time[1] << std::endl;
  }
  std::cout << "Sumfactored Laplace was " << time[1]/time[0] << " times faster!" << std::endl;

  // Test with ipdg energy norm:
  double error;
  {
    Ax-=Ax_mf;
    auto ipdg = Dune::Fufem::MatrixFree::IPDGNorm<Vector, GV, decltype(basis)>(basis);
    auto ipdgop = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(ipdg)>(gv, ipdg);

    auto dummy= Ax;
    ipdgop.apply(Ax, dummy);
    error = Ax*dummy;
  }
  std::cout << error << std::endl;
  suite.check(error<1e-12, "Check if sumfacttorized and standard matrix free Laplace yield the same") << "Difference for order " <<k <<" is " << error << std::endl;

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{16,16}});
  TestSuite suite;
  for (int k=1; k < 7; k++) {
    suite.subTest(test_bulk(grid.leafGridView(), k));
    suite.subTest(test_IPDG(grid.leafGridView(), k));
    std::cout << "-------------------------------------------" << std::endl;
  }
  return suite.exit();
}
