// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions
#include <dune/common/test/testsuite.hh> // We use exceptions

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/functions/functionspacebases/pq1nodalbasis.hh>

#include <dune/functions/functionspacebases/gridadaptor.hh>
#include <dune/functions/functionspacebases/hierarchicvectorwrapper.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <dune/istl/bvector.hh>
using namespace Dune;

auto checkGridAdaptor() {
  TestSuite suite;
  // Grid
  using GridType =Dune::YaspGrid<2>;
  auto grid = Dune::StructuredGridFactory<GridType>
    ::createCubeGrid({{0,0}}, {{1,1}}, {{2,2}});

  // Basis
  using Basis = Dune::Functions::PQ1NodalBasis<GridType::LeafGridView>;
  Basis basis(grid->leafGridView());

  Functions::Impl::PersistentBasis<Basis> persistentBasis(basis);

  // setup some coefficient vector
  using Vector = BlockVector<FieldVector<double,1>>;
  Vector cc;
  cc.resize(basis.size());
  auto ccbe = Functions::hierarchicVector(cc);
  Dune::Functions::interpolateTree(basis, TypeTree::hybridTreePath(), ccbe,[&](const auto& x) {return x[0];});

  // adapt grid once. Any data belonging to the basis in its old state
  // will be invalid now.
  auto gridAdaptor = Functions::GridAdaptor<Basis>(basis);

  grid->globalRefine(1);
  basis.update(grid->leafGridView());

  persistentBasis.update();
  for (const auto& e : elements(grid->leafGridView())) {
    // all elements should be new
    suite.check(persistentBasis.isNewElement(e), "Check if elements are recognized as new");
    // but their fathers not
    suite.check(not persistentBasis.isNewElement(e.father()), "Check if father of new elements are known");
  }


  // right now, the fine vector has to be manually resized to the correct format
  auto fc = cc;
  fc.resize(basis.size());
  auto fcbe = Functions::hierarchicVector(fc);
  gridAdaptor.adapt(ccbe, fcbe);

  std::cout << fc << std::endl;
  return suite;
}

int main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  TestSuite suite;

  suite.subTest(checkGridAdaptor());
  return suite.exit();
}
