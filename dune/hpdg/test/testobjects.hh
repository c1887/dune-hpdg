#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/dunefunctionsfunctionalassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/l2functionalassembler.hh>
#include <dune/fufem/functions/constantfunction.hh>

#include <dune/fufem/assemblers/localassemblers/interiorpenaltydgassembler.hh>
#include <dune/hpdg/functionspacebases/dgqkglbasis.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>

/** Assemble a stiffness matrix */
template<int k, class GridType>
auto stiffnessMatrix(const GridType& grid, double penaltyFactor=1.5) {
  constexpr auto dim = GridType::dimensionworld;

  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, Dune::StaticPower<k+1, dim>::power, Dune::StaticPower<k+1, dim>::power> >;
  Matrix matrix;

  const auto penalty = penaltyFactor*std::pow((double) k, dim); // penalty factor

  /* Setup Basis */
  using Basis = Dune::Functions::DGQkGLBlockBasis<typename GridType::LeafGridView, k>;
  Basis basis{grid.leafGridView()};

  /* assemble laplace bulk terms and ipdg terms */
  {
    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix);
    auto patternBuilder = matrixBackend.patternBuilder();

    auto assembler = Assembler{basis, basis};

    assembler.assembleSkeletonPattern(patternBuilder);

    patternBuilder.setupMatrix();


    auto vintageIPDGAssembler = InteriorPenaltyDGAssembler<GridType, FiniteElement, FiniteElement>();
    vintageIPDGAssembler.sigma0=penalty;
    vintageIPDGAssembler.dirichlet = true;
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

    /* We need to construct the stiffness Matrix into a separate (temporary) matrix object as otherwise the previously assembled entries
     * would be lost. This temporary matrix will be deleted after we leave the block scope*/
    auto bulkMatrix = Matrix{};
    auto bmatrixBackend = Dune::Fufem::istlMatrixBackend(bulkMatrix);
    assembler.assembleBulk(bmatrixBackend, vintageBulkAssembler);
    matrix+=bulkMatrix;
  }

  return matrix;
}

/** Assemble a dynamic stiffness matrix */
template<class GridType>
auto dynamicStiffnessMatrix(const GridType& grid, int k, double penaltyFactor=1.5) {
  constexpr auto dim = GridType::dimensionworld;

  const auto penalty = penaltyFactor*std::pow((double) k, dim); // penalty factor

  /* Setup Basis */
  using Basis = Dune::Functions::DynamicDGQkGLBlockBasis<typename GridType::LeafGridView>;
  Basis basis{grid.leafGridView(), k};
  auto blockSize = (size_t) std::pow((size_t) k+1, (size_t)dim);

  /* assemble laplace bulk terms and ipdg terms */
  using DynBCRS = Dune::HPDG::DynamicBCRSMatrix<double>;
  DynBCRS dynMatrix{};
  auto& matrix = dynMatrix.matrix();
  {
    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix);
    auto patternBuilder = matrixBackend.patternBuilder();

    auto assembler = Assembler{basis, basis};

    assembler.assembleSkeletonPattern(patternBuilder);

    patternBuilder.setupMatrix();
    dynMatrix.finishIdx();
    for (size_t i = 0; i <matrix.N(); i++)
      dynMatrix.blockRows(i) = blockSize;
    dynMatrix.setSquare();
    dynMatrix.update();

    auto vintageIPDGAssembler = InteriorPenaltyDGAssembler<GridType, FiniteElement, FiniteElement>();
    vintageIPDGAssembler.sigma0=penalty;
    vintageIPDGAssembler.dirichlet = true;
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
    bulkMatrix.matrix()=0;
    auto bmatrixBackend = Dune::Fufem::istlMatrixBackend(bulkMatrix.matrix());
    // bulk pattern was inherited from the skeleton part
    assembler.assembleBulkEntries(bmatrixBackend, vintageBulkAssembler);
    matrix+=bulkMatrix.matrix();
  }

  return dynMatrix;
}


template<int k, class GridType>
auto rightHandSide(const GridType& grid, double force=-10.0) {
  constexpr auto dim = GridType::dimensionworld;
  using Vector = Dune::BlockVector<Dune::FieldVector<double, Dune::StaticPower<k+1, dim>::power> >;
  Vector rhs(grid.leafGridView().size(0));

  using Basis = Dune::Functions::DGQkGLBlockBasis<typename GridType::LeafGridView, k>;
  Basis basis{grid.leafGridView()};
  auto rhsBE = Dune::Fufem::istlVectorBackend(rhs);

  using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;

  // assemble standard function \int fv
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

template<class GridType>
auto dynamicRightHandSide(const GridType& grid, int k=1, double force=-10.0) {
  constexpr auto dim = GridType::dimensionworld;
  //using Vector = Dune::BlockVector<Dune::FieldVector<double, Dune::StaticPower<k+1, dim>::power> >;
  using Vector = Dune::HPDG::DynamicBlockVector<double>;
  int blockSize = (int) std::pow(k+1, (int) dim);
  Vector rhs(grid.leafGridView().size(0), blockSize);
  rhs.update();

  //using Basis = Dune::Functions::DGQkGLBlockBasis<typename GridType::LeafGridView, k>;
  //Basis basis{grid.leafGridView()};

  using Basis = Dune::Functions::DynamicDGQkGLBlockBasis<typename GridType::LeafGridView>;
  Basis basis{grid.leafGridView(), k};

  auto rhsBE = Dune::Fufem::istlVectorBackend(rhs);

  using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;

  // assemble standard function \int fv
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
