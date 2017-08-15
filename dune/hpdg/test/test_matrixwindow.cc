#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/common/matrixwindow.hh>
#include <dune/istl/io.hh>
#include <dune/istl/matrixindexset.hh>
#include <memory>

using namespace Dune;



TestSuite test_matrixwindow() {
  TestSuite suite("test_matrixwindow");

  const size_t n = 5;
  const size_t m= 5;
  auto data = std::make_unique<double[]>(n*m);
  Dune::HPDG::MatrixWindow<double> mat{data.get(), n,m};
  // test empty constructor
  {
    Dune::HPDG::MatrixWindow<double> zeromat{};
  }

  // we can set things
  mat[3][2] = 1.0;

  //auto print = [&](const auto& mat) {
  //for (size_t i = 0; i < mat.N(); i++) {
    //for (size_t j = 0; j < mat.M(); j++) {
      //std::cout << mat[i][j] << " ";
    //}
    //std::cout << std::endl;
  //}
  //};


  // scalar assignment:
  mat=4.2;

  // we even have iterators, whoop whoop
  for (const auto& row: mat) {
    for (const auto& col: row) {
      suite.check(col==4.2);
    }
  }

  //print(mat);

  // matrix-vector:
  auto fv = Dune::FieldVector<double, m>();
  fv=3.;
  {
    auto dummy = fv;
    mat.mv(dummy, fv);
  }

  for (const auto& e: fv) suite.check(std::abs(e- 4.2*3*m)<1e-15);


  // lets see, if we can create a bcrs matrix
  using Matrix = Dune::BCRSMatrix<Dune::HPDG::MatrixWindow<double> >;
  Matrix bcrs{};

  Dune::MatrixIndexSet idx(2,2);
  idx.add(0,0);
  idx.add(1,0);
  idx.add(1,1);
  idx.exportIdx(bcrs);

  // here, normally one would set different pointers for every block (well, maybe not, if you're sure they're equal this would be quite nice)
  bcrs[0][0].set(data.get(), n,m);
  bcrs[1][1].set(data.get(), n,m);
  bcrs[1][0].set(data.get(), n,m);

  using BV = Dune::BlockVector<Dune::FieldVector<double, m>>;
  BV bv(2);
  bv=2.;
  {
    auto dummy = bv;
    bcrs.mv(dummy, bv);
  }

  //for (const auto& e: bv) std::cout << e << std::endl;
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_matrixwindow());
  return suite.exit();
}
