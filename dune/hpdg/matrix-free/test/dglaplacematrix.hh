#pragma once

#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/dunefunctionsfunctionalassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/l2functionalassembler.hh>
#include <dune/fufem/functions/constantfunction.hh>

#include <dune/fufem/assemblers/localassemblers/interiorpenaltydgassembler.hh>

#include <dune/functions/backends/istlvectorbackend.hh>

namespace Dune {
namespace ParMG {

/** Assemble a stiffness matrix */
template<typename Basis>
auto stiffnessMatrix(const Basis& basis, double penalty=10.0) {
  using GridType = typename Basis::GridView::Grid;

  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  Matrix matrix;

  /* assemble laplace bulk terms and ipdg terms */
  {
    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix);
    auto patternBuilder = matrixBackend.patternBuilder();

    auto assembler = Assembler{basis, basis};

    assembler.assembleSkeletonPattern(patternBuilder);

    patternBuilder.setupMatrix();


    auto vintageIPDGAssembler = InteriorPenaltyDGAssembler<GridType, FiniteElement, FiniteElement>(penalty);
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

    auto vintageBulkAssembler = LaplaceAssembler<GridType,FiniteElement, FiniteElement>();

     //We need to construct the stiffness Matrix into a separate (temporary) matrix object as otherwise the previously assembled entries
     //would be lost. This temporary matrix will be deleted after we leave the block scope
    auto bulkMatrix = Matrix{};
    auto bmatrixBackend = Dune::Fufem::istlMatrixBackend(bulkMatrix);
    assembler.assembleBulk(bmatrixBackend, vintageBulkAssembler);
    matrix+=bulkMatrix;
  }

  return matrix;
}

template<typename Basis>
auto rightHandSide(const Basis& basis, double force=-10.0) {
  using GridType = typename Basis::GridView::Grid;
  constexpr auto dim = GridType::dimensionworld;

  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  Vector rhs(basis.dimension());

  auto rhsBE = Dune::Functions::istlVectorBackend(rhs);

  using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;

  {
    const ConstantFunction<Dune::FieldVector<double, dim>, Dune::FieldVector<double, 1> > f(force);
    const L2FunctionalAssembler<GridType, FiniteElement> rhsLocalAssembler(f);
    const auto localRHSlambda = [&](const auto& element, auto& localV, const auto& localView) {
      rhsLocalAssembler.assemble(element, localV, localView.tree().finiteElement());
    };
    Dune::Fufem::DuneFunctionsFunctionalAssembler<Basis> rhsAssembler(basis);
    rhsAssembler.assembleBulk(rhsBE, localRHSlambda);
  }

  return rhs;
}
} /* namespace ParMG */
} /* namespace Dune */
