// TODO: Aufräumen!
#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/common/dynmatrix.hh>

#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkgausslegendrebasis.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkgausskronrodbasis.hh>
#include <dune/hpdg/transferoperators/dynamicordertransfer.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/resizehelper.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>

using namespace Dune;

template<class Basis, class Grid>
TestSuite test_dynamicBasis(Basis& basis, const Grid& grid) {
  TestSuite suite;


  auto lv = basis.localView();
  auto li = basis.localIndexSet();
  // set one arbitrary node to have degree 2 (all others have degree 1)
  for (const auto& e: elements(grid.leafGridView())) {
    basis.preBasis().degree(e) = 2;
    break;
  }

  // check dimension: should be (#elements-1)*4+ 9 as all but one elements have 4 DoFs and the other one has 9
  suite.check(basis.dimension() == (size_t) (grid.leafGridView().size(0)-1)*4 +9, "Check if dimension was calculated correctly");

  // now, assemble a matrix
  using Matrix = HPDG::DynamicBCRSMatrix<double>;
  auto matrix = Matrix{};
  using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
  auto assembler = Assembler{basis, basis};
  using FE = std::decay_t<decltype(lv.tree().finiteElement())>;
  auto laplace = LaplaceAssembler<Grid, FE, FE>{};

  // set matrix pattern
  auto mBE = Dune::Fufem::istlMatrixBackend(matrix.matrix());
  { // setup pattern
    auto pb = mBE.patternBuilder();
    assembler.assembleBulkPattern(pb);
    pb.setupMatrix();
  }

  // set block sizes and allocate memory:
  Dune::HPDG::resizeFromBasis(matrix, basis);
  matrix.update();

  // now, assemble:
  assembler.assembleBulkEntries(mBE, laplace);

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  using Grid = YaspGrid<2>;
  auto grid = StructuredGridFactory<Grid>::createCubeGrid({0,0}, {1.0,1.0}, {{4,4}});


  TestSuite suite;

  auto gaussLobattoBasis = Dune::Functions::DynamicDGQkGLBlockBasis<Grid::LeafGridView>{grid->leafGridView(), 1};
  suite.subTest(test_dynamicBasis(gaussLobattoBasis, *grid));

  auto gaussLegendreBasis = Dune::Functions::DynamicDGQkGaussLegendreBlockBasis<Grid::LeafGridView>{grid->leafGridView(), 1};
  suite.subTest(test_dynamicBasis(gaussLegendreBasis, *grid));

  auto gaussKronrodBasis = Dune::Functions::DynamicDGQkGaussKronrodBlockBasis<Grid::LeafGridView>{grid->leafGridView(), 2};
  suite.subTest(test_dynamicBasis(gaussLegendreBasis, *grid));

  return suite.exit();
}
