#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/common/dynmatrix.hh>

#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>

#include <dune/matrix-vector/algorithm.hh>
using namespace Dune;

TestSuite test_makedgtransfertuple() {
  TestSuite suite;

  using Grid = YaspGrid<2>;
  auto grid = StructuredGridFactory<Grid>::createCubeGrid({0,0}, {1.0,1.0}, {{4,4}});

  auto basis = Dune::Functions::DynamicDGQkGLBlockBasis<Grid::LeafGridView>{grid->leafGridView(), 1};
  using Basis = decltype(basis);

  auto lv = basis.localView();
  auto li = basis.localIndexSet();
  // set one arbitrary node to have degree 2 (all others have degree 1)
  for (const auto& e: elements(grid->leafGridView())) {
    basis.nodeFactory().degree(e) = 2;
    break;
  }

  // check dimension: should be (#elements-1)*4+ 9 as all but one elements have 4 DoFs and the other one has 9
  suite.check(basis.dimension() == (size_t) (grid->leafGridView().size(0)-1)*4 +9, "Check if dimension was calculated correctly");

  // now, assemble a matrix
  using Matrix = BCRSMatrix<DynamicMatrix<double>>;
  auto matrix = Matrix{};
  using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
  auto assembler = Assembler{basis, basis};
  using FE = std::decay_t<decltype(lv.tree().finiteElement())>;
  auto laplace = LaplaceAssembler<Grid, FE, FE>{};

  auto mBE = Dune::Fufem::istlMatrixBackend(matrix);
  { // setup pattern
    auto pb = mBE.patternBuilder();
    assembler.assembleBulkPattern(pb);
    pb.setupMatrix();
  }

  // now comes the ugly part, setting the sizes of the local blocks.
  for (const auto& e: elements(grid->leafGridView())) {
    lv.bind(e);
    li.bind(lv);
    auto i = li.index(0)[0]; // element index in this basis
    auto n = lv.size();
    matrix[i][i].resize(n,n);

    // the following is only needed if also intersections are assembled, the bulk is on one element for DG
    /*
    for (const auto& is: intersections(grid->leafGridView(), e)) {
      if (not is.neighbor()) continue;
      const auto& out = is.outside();
      lv.bind(out);
      li.bind(lv);
      auto j = li.index(0)[0]; // element index in this basis
      auto m = lv.size();
      if (not matrix.exists(i,j)) DUNE_THROW(Dune::Exception, "Yo, no such entry");
      matrix[i][j].resize(n,m);
    }
    */
  }

  // now, assemble:
  assembler.assembleBulkEntries(mBE, laplace);

  // print two blocks:
  //for (size_t idx = 0; idx<2; idx++) {
    //std::cout << "Block " << idx <<", " << idx <<std::endl;
    //const auto& mii = matrix[idx][idx];
    //for (size_t i = 0; i <mii.N(); i++) {
      //for (size_t j = 0; j <mii.M(); j++) {
        //std::cout << mii[i][j] << " ";
      //}
      //std::cout << std::endl;
    //}
  //}
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_makedgtransfertuple());
  return suite.exit();
}
