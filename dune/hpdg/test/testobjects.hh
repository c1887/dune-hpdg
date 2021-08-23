#pragma once
#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/dunefunctionsfunctionalassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>
#include <dune/fufem/assemblers/localassemblers/dunefunctionsl2functionalassembler.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/fufem/assemblers/localassemblers/interiorpenaltydgassembler.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

#include <dune/functions/backends/istlvectorbackend.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>

/** Assemble a dynamic stiffness matrix */
template<class GridType>
auto dynamicStiffnessMatrix(const GridType& grid, int k, double penaltyFactor=1.5, bool dirichlet=true) {
  constexpr auto dim = GridType::dimensionworld;

  const auto penalty = penaltyFactor*std::pow((double) k, dim); // penalty factor

  /* Setup Basis */
  using Basis = Dune::Functions::DynamicDGQkGLBlockBasis<typename GridType::LeafGridView>;
  Basis basis{grid.leafGridView(), k};
  auto blockSize = (size_t) std::pow((size_t) k+1, (size_t)dim);

  /* assemble laplace bulk terms and ipdg terms */
  using DynBCRS = Dune::HPDG::DynamicBCRSMatrix<Dune::FieldMatrix<double, 1,1>>;
  DynBCRS dynMatrix{};
  auto& matrix = dynMatrix;
  {
    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix.asBCRSMatrix());
    auto patternBuilder = matrixBackend.patternBuilder();

    auto assembler = Assembler{basis, basis};

    assembler.assembleSkeletonPattern(patternBuilder);

    patternBuilder.setupMatrix();
    dynMatrix.finishIdx();
    for (size_t i = 0; i <matrix.N(); i++)
      dynMatrix.blockRows(i) = blockSize;
    dynMatrix.setSquare();
    dynMatrix.update();

    auto vintageIPDGAssembler = InteriorPenaltyDGAssembler<GridType, FiniteElement, FiniteElement>(penalty, dirichlet);
    auto localBlockAssembler = [&](const auto& edge, auto& matrixContainer,
        auto&& insideTrialLocalView, auto&& insideAnsatzLocalView, auto&& outsideTrialLocalView, auto&& outsideAnsatzLocalView)
    {
        vintageIPDGAssembler.assembleBlockwise(edge, matrixContainer, insideTrialLocalView.tree().finiteElement(),
                                               insideAnsatzLocalView.tree().finiteElement(),
                                               outsideTrialLocalView.tree().finiteElement(),
                                               outsideAnsatzLocalView.tree().finiteElement());
    };
    auto localBoundaryAssembler = [&](const auto& edge, auto& localMatrix, auto&& insideTrialLocalView, auto&& insideAnsatzLocalView)
    {
        vintageIPDGAssembler.assemble(edge, localMatrix, insideTrialLocalView.tree().finiteElement(), insideAnsatzLocalView.tree().finiteElement());
    };

    assembler.assembleSkeletonEntries(matrixBackend, localBlockAssembler, localBoundaryAssembler); // IPDG terms

    //auto vintageBulkAssembler = Dune::Fufem::ConstantLaplaceAssembler<GridType,FiniteElement, FiniteElement>();
    auto vintageBulkAssembler = LaplaceAssembler<GridType,FiniteElement, FiniteElement>();

    /* We need to construct the stiffness Matrix into a separate (temporary) matrix object as otherwise the previously assembled entries
     * would be lost. This temporary matrix will be deleted after we leave the block scope*/
    auto bulkMatrix = dynMatrix;
    bulkMatrix=0;
    auto bmatrixBackend = Dune::Fufem::istlMatrixBackend(bulkMatrix);
    // bulk pattern was inherited from the skeleton part
    assembler.assembleBulkEntries(bmatrixBackend, vintageBulkAssembler);
    matrix+=bulkMatrix;
  }

  return dynMatrix;
}


/** Assemble a dynamic stiffness matrix */
template<class GridType>
auto dynamicMassMatrix(const GridType& grid, int k) {
  constexpr auto dim = GridType::dimensionworld;

  /* Setup Basis */
  using Basis = Dune::Functions::DynamicDGQkGLBlockBasis<typename GridType::LeafGridView>;
  Basis basis{grid.leafGridView(), k};
  auto blockSize = (size_t) std::pow((size_t) k+1, (size_t)dim);

  /* assemble laplace bulk terms and ipdg terms */
  using DynBCRS = Dune::HPDG::DynamicBCRSMatrix<Dune::FieldMatrix<double, 1,1>>;
  DynBCRS dynMatrix{};
  auto& matrix = dynMatrix;
  {
    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix.asBCRSMatrix());
    auto patternBuilder = matrixBackend.patternBuilder();

    auto assembler = Assembler{basis, basis};

    assembler.assembleBulkPattern(patternBuilder);

    patternBuilder.setupMatrix();
    dynMatrix.finishIdx();
    for (size_t i = 0; i <matrix.N(); i++)
      dynMatrix.blockRows(i) = blockSize;
    dynMatrix.setSquare();
    dynMatrix.update();

    auto vintageBulkAssembler = MassAssembler<GridType,FiniteElement, FiniteElement>();

    dynMatrix=0.;
    assembler.assembleBulkEntries(matrixBackend, vintageBulkAssembler);
  }

  return dynMatrix;
}

template<class GridType>
auto dynamicRightHandSide(const GridType& grid, int k=1, double force=-10.0) {
  constexpr auto dim = GridType::dimensionworld;
  //using Vector = Dune::BlockVector<Dune::FieldVector<double, Dune::StaticPower<k+1, dim>::power> >;
  using Vector = Dune::HPDG::DynamicBlockVector<Dune::FieldVector<double, 1>>;
  int blockSize = (int) std::pow(k+1, (int) dim);
  Vector rhs(grid.leafGridView().size(0), blockSize);
  rhs.update();

  //using Basis = Dune::Functions::DGQkGLBlockBasis<typename GridType::LeafGridView, k>;
  //Basis basis{grid.leafGridView()};

  using Basis = Dune::Functions::DynamicDGQkGLBlockBasis<typename GridType::LeafGridView>;
  Basis basis{grid.leafGridView(), k};

  auto rhsBE = Dune::Functions::istlVectorBackend(rhs);

  using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;

  // assemble standard function \int fv
  {
    auto f = [&force] (const auto&) { return Dune::FieldVector<double, 1>(force); };
    auto ff = Dune::Functions::makeAnalyticGridViewFunction(f, basis.gridView());
    auto key = QuadratureRuleKey(dim, 0);
    auto rhsLocalAssembler = Dune::Fufem::DuneFunctionsL2FunctionalAssembler<GridType, FiniteElement, decltype(ff)>{ff, key};
    const auto localRHSlambda = [&](const auto& element, auto& localV, const auto& localView) {
      rhsLocalAssembler.assemble(element, localV, localView.tree().finiteElement());
    };
    Dune::Fufem::DuneFunctionsFunctionalAssembler<Basis> rhsAssembler(basis);
    // We only have the correct vector size, hence we only need to assemble the Entries
    rhsAssembler.assembleBulkEntries(rhsBE, localRHSlambda);
  }

  return rhs;
}
