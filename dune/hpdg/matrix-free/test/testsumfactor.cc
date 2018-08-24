#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>


#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/sflaplace.hh>
#include <dune/hpdg/matrix-free/localoperators/uniformlaplaceoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/laplaceoperator.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/common/resizehelper.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
using namespace Dune;

// Tests if sumfactorized and standard matrix-free bulk laplace yield the same vectors
template<class GV>
TestSuite test_bulk(const GV& gv, int k) {
  TestSuite suite;
  using Vector = Dune::HPDG::DynamicBlockVector<double>;

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

  Ax-=Ax_mf;
  //auto error = Ax.two_norm();
  auto error = Ax[0]*Ax[0];
  std::cout << error << std::endl;
  suite.check(error<1e-12, "Check if sumfacttorized and standard matrix free Laplace yield the same") << "Difference for order " <<k <<" is " << error << std::endl;

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{16,16}});
  TestSuite suite;
  for (int k=1; k < 9; k++)
    suite.subTest(test_bulk(grid.leafGridView(), k));
  return suite.exit();
}
