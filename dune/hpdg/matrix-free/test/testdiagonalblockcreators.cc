#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/hpdg/test/testobjects.hh>

#include <dune/hpdg/matrix-free/localoperators/ipdgdiagonalblock.hh>
#include <dune/hpdg/matrix-free/localoperators/heatdiagonalblock.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>

#include <dune/fufem/assemblers/localassemblers/massassembler.hh>

using namespace Dune;

// test the fast assembly of a diagonal block of the form
// a(\nabla \phi_i, \nabla \phi_j) + b (\phi_i, \phi_j)
//
// where a and b are scalar factors
template<class GV>
TestSuite test_heat_diag(const GV& gv) {
  TestSuite suite;

  constexpr int order = 3;
  auto penalty = 2.0;
  using Basis = Functions::DynamicDGQkGLBlockBasis<GV>;
  Basis basis(gv, order);
  auto lv = basis.localView();

  // add some factors to spice things up
  auto laplace_factor = 0.3;
  auto mass_factor = 0.7;

  auto fully_assembled_ipdg = dynamicStiffnessMatrix(gv.grid(), order, penalty, false);
  fully_assembled_ipdg.matrix() *= laplace_factor;
  // add also some mass term:
  {

    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;

    auto assembler = Assembler{basis, basis};

    using FiniteElement = std::decay_t<decltype(lv.tree().finiteElement())>;
    using Grid = typename GV::Grid;
    auto mass = fully_assembled_ipdg;
    mass =0.;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(mass.matrix());

    auto massA = MassAssembler<Grid, FiniteElement, FiniteElement>();

    assembler.assembleBulkEntries(matrixBackend, massA);

    mass.matrix() *= mass_factor;

    fully_assembled_ipdg.matrix() += mass.matrix();
  }

  auto matrixCreator = HPDG::HeatDiagonalBlock<Basis>(basis, penalty, mass_factor, laplace_factor);

  for(const auto& element : elements(gv)) {
    matrixCreator.bind(element);
    lv.bind(element);

    auto matrix = matrixCreator.matrix(); // get diagonal block wrt to the given element

    auto i = lv.index(0)[0]; // get element index in basis
    const auto& assembledDiagonalBlock = fully_assembled_ipdg[i][i];

    matrix -= assembledDiagonalBlock;
    suite.check(matrix.frobenius_norm() < 1e-13, "Matrix blocks do not match") << "Error: " << matrix.frobenius_norm();
  }

  return suite;
}

template<class GV>
TestSuite test_ipdg_diag(const GV& gv) {
  TestSuite suite;

  constexpr int order = 3;
  auto penalty = 2.0;
  using Basis = Functions::DynamicDGQkGLBlockBasis<GV>;
  Basis basis(gv, order);
  auto lv = basis.localView();

  auto fully_assembled_ipdg = dynamicStiffnessMatrix(gv.grid(), order, penalty);

  auto matrixCreator = HPDG::IPDGDiagonalBlock<Basis>(basis, penalty, true);


  for(const auto& element : elements(gv)) {
    matrixCreator.bind(element);
    lv.bind(element);

    auto matrix = matrixCreator.matrix(); // get diagonal block wrt to the given element

    auto i = lv.index(0)[0]; // get element index in basis
    const auto& assembledDiagonalBlock = fully_assembled_ipdg[i][i];

    matrix -= assembledDiagonalBlock;
    suite.check(matrix.frobenius_norm() < 1e-13, "Matrix blocks do not match") << "Error: " << matrix.frobenius_norm();
  }

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{8,8}});
  TestSuite suite;
  suite.subTest(test_ipdg_diag(grid.leafGridView()));
  suite.subTest(test_heat_diag(grid.leafGridView()));
  return suite.exit();
}
