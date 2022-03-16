// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_MATRIX_FREE_LOCAL_WEIGHTED_IPDG_BLOCK_JACOBI_HH
#define DUNE_FUFEM_MATRIX_FREE_LOCAL_WEIGHTED_IPDG_BLOCK_JACOBI_HH
#include <dune/common/fmatrix.hh>
#include <dune/common/math.hh>

#include <dune/istl/matrix.hh>

#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/hpdg/assemblers/localassemblers/variableipdg.hh>
#include <dune/fufem/quadraturerules/quadraturerulecache.hh>
#include <dune/functions/backends/istlvectorbackend.hh>

#include <dune/istl/matrix.hh>

#include <dune/hpdg/assemblers/localassemblers/dunefunctionsweightedmassassembler.hh>
#include <dune/hpdg/common/indexedcache.hh>
#include <dune/hpdg/common/mutablequadraturenodes.hh>
#include <dune/hpdg/localfunctions/assemblycache.hh>
#include <dune/hpdg/matrix-free/localoperators/gausslobattomatrices.hh>

/** This is a factory for the diagonal block of an IPDG Laplace discretization.
 * After binding to an element via bind(), you can get the corresponding
 * diagonal matrix block of the stiffness matrix via matrix().
 *
 * This is particulary useful for block Jacobi algorithms.
 */
namespace Dune {
namespace HPDG {

template<class Basis, class GradientWeight, class MassWeight>
class WeightedIPDGHeatDiagonalBlock
{
  using GV = typename Basis::GridView;
  static constexpr int dim = GV::dimension;
  using LV = typename Basis::LocalView;
  using Field = double;
  using FV = Dune::FieldVector<Field, dim>;
  using LocalMatrix =
    Dune::Matrix<Dune::FieldMatrix<Field, 1, 1>>; // TODO: This is not
                                                  // necessarily 1x1
  using FE = std::decay_t<decltype(std::declval<LV>().tree().finiteElement())>;

public:
  WeightedIPDGHeatDiagonalBlock(const Basis& b,
                                const GradientWeight& gw,
                                const MassWeight& mw,
                                int extraQuadOrder,
                                double penalty = 2.0,
                                bool dirichlet = false)
    : basis_(b)
    , gradWeight_(gw)
    , massWeight_(mw)
    , extraQuadOrder_(extraQuadOrder)
    , penalty_(penalty)
    , dirichlet_(dirichlet)
    , localView_(basis_.localView())
    , outerView_(basis_.localView())
  {}

  template<class Entity>
  void bind(const Entity& e)
  {
    localView_.bind(e);

    localMatrix_.setSize(localView_.size(), localView_.size());
    localMatrix_ = 0.;
  }

  /** Get the matrix for the bound element */
  const auto& matrix()
  {
    compute();
    return localMatrix_;
  }

  void setFactors(double laplace, double mass) {
    factors_.laplace = laplace;
    factors_.mass = mass;
  }

private:
  // compute matrix diagonal block
  // and solve block for input part
  void compute()
  {
    computeBulk();
    computeFace();
  }

  void computeFace()
  {
    const auto& fe = localView_.tree().finiteElement();
    insideCache_.bind(&fe);

    for (const auto& is :
         intersections(basis_.gridView(), localView_.element())) {
      if (not is.neighbor()) {
        // TODO Das funktioniert hier irgendwie nicht. Tun wir erstmal so, als
        // sei das Neumann
        // continue; // TODO <-- Die Zeile dann löschen
        if (dirichlet_)
          DUNE_THROW(Dune::NotImplemented, "no dirichlet yet");
        continue;
        // auto tmp =localMatrix_;

        // localAssembler.assemble(is, tmp, insideCache_, insideCache_);
        // //localMatrix_+=tmp;
        // for(std::size_t i = 0; i < fe.size(); i++) {
        //   auto row = localView_.index(i)[1];
        //   for(std::size_t j = 0; j < fe.size(); j++) {
        //     auto col = localView_.index(j)[1];
        //     localMatrix_[row][col]+=tmp[i][j];
        //   }
        // }
      } else {
        const auto& outside = is.outside();
        outerView_.bind(outside);
        const auto& ofe = outerView_.tree().finiteElement();

        const auto maxOrder =
          std::max(ofe.localBasis().order(), fe.localBasis().order());
        const auto penalty = penalty_ * maxOrder *
                       maxOrder; // TODO: ^2 scaling is wrong for dim==1

        using FVdimworld = typename Dune::template FieldVector<double, dim>;

        using JacobianType =
          typename FE::Traits::LocalBasisType::Traits::JacobianType;
        using RangeType =
          typename FE::Traits::LocalBasisType::Traits::RangeType;

        // get geometry and store it
        const auto& edgeGeometry = is.geometry();
        const auto& insideGeometry = is.inside().geometry();

        const auto edgeLength = edgeGeometry.volume();

        // get quadrature rule
        QuadratureRuleKey tFEquad(is.type(), maxOrder);
        QuadratureRuleKey quadKey = tFEquad.square().square();

        const auto& quad = QuadratureRuleCache<double, dim - 1>::rule(quadKey);

        // store gradients for both the inner and outer elements
        std::vector<JacobianType> insideReferenceGradients(
          fe.localBasis().size());
        std::vector<FVdimworld> insideGradients(fe.localBasis().size());

        // store values of shape functions
        std::vector<RangeType> tFEinsideValues(fe.localBasis().size());

        const auto outerNormal = is.centerUnitOuterNormal();

        auto lf = localFunction(gradWeight_);
        lf.bind(localView_.element());

        // loop over quadrature points
        for (size_t pt = 0; pt < quad.size(); ++pt) {
          // get quadrature point
          const auto& quadPos = quad[pt].position();

          const auto weight = lf(is.geometryInInside().global(quadPos));
          assert(not std::isnan(weight));

          // get transposed inverse of Jacobian of transformation
          const auto& invJacobian = insideGeometry.jacobianInverseTransposed(
            is.geometryInInside().global(quadPos));

          // get integration factor
          const auto integrationElement =
            edgeGeometry.integrationElement(quadPos);

          // get gradients of shape functions on both the inside and outside
          // element
          insideCache_.localBasis().evaluateJacobian(
            is.geometryInInside().global(quadPos), insideReferenceGradients);

          // transform gradients
          for (size_t i = 0; i < insideGradients.size(); ++i)
            invJacobian.mv(insideReferenceGradients[i][0], insideGradients[i]);

          // evaluate basis functions
          insideCache_.localBasis().evaluateFunction(
            is.geometryInInside().global(quadPos), tFEinsideValues);

          // compute matrix entries
          auto z = quad[pt].weight() * integrationElement;

          // Basis functions from inside as test functions
          for (size_t i = 0; i < fe.localBasis().size(); ++i) {
            // Basis functions from inside as ansatz functions
            for (size_t j = 0; j < fe.localBasis().size(); ++j) {
              // M11, see Riviere, p. 54f
              auto zij = -0.5 * z * tFEinsideValues[i] *
                         (weight * (insideGradients[j] * outerNormal));
              zij += -0.5 * z * tFEinsideValues[j] * weight *
                     (insideGradients[i] * outerNormal);
              zij += penalty * std::max(1.0, weight) * z / edgeLength * tFEinsideValues[i] *
                     tFEinsideValues[j];
              localMatrix_[i][j] += factors_.laplace * zij;
            }
          }
        }

        /*
        for(std::size_t i = 0; i < fe.size(); i++) {
          auto row = localView_.index(i)[1];
          for(std::size_t j = 0; j < fe.size(); j++) {
            auto col = localView_.index(j)[1];
            localMatrix_[row][col]+=tmp[i][j];
          }
        }
        */
      }
    }
  }

  /* TODO: Put this also into dune-fufem */
  auto laplaceBlock()
  {

    using JacobianType =
      typename FE::Traits::LocalBasisType::Traits::JacobianType;
    using FVdimworld = typename Dune::template FieldVector<double, dim>;

    const auto& fe = localView_.tree().finiteElement();
    insideCache_.bind(&fe);
    // using FE = std::decay_t<decltype(fe)>;
    // auto localAssembler = LaplaceAssembler<typename GV::Grid,
    // HPDG::AssemblyCache<FE>, HPDG::AssemblyCache<FE>>();
    auto quadKey =
      QuadratureRuleKey(localView_.element().type(), extraQuadOrder_);
    auto localMatrix = localMatrix_;
    localMatrix = 0.0;

    QuadratureRuleKey tFEquad(fe);
    QuadratureRuleKey quadKeyFE =
      tFEquad.derivative().square().product(quadKey);
    const Dune::template QuadratureRule<double, dim>& quad =
      QuadratureRuleCache<double, dim>::rule(quadKeyFE);

    // store gradients of shape functions and base functions
    std::vector<JacobianType> referenceGradients(fe.localBasis().size());
    std::vector<FVdimworld> gradients(fe.localBasis().size());

    auto lf = localFunction(gradWeight_);
    lf.bind(localView_.element());

    const auto& geometry = localView_.element().geometry();

    // loop over quadrature points
    for (size_t pt = 0; pt < quad.size(); ++pt) {
      // get quadrature point
      const auto& quadPos = quad[pt].position();

      // get transposed inverse of Jacobian of transformation
      const auto& invJacobian =
        geometry.jacobianInverseTransposed(quadPos);

      // get integration factor
      const double integrationElement = geometry.integrationElement(quadPos);

      // get gradients of shape functions
      fe.localBasis().evaluateJacobian(quadPos, referenceGradients);

      // transform gradients
      for (size_t i = 0; i < gradients.size(); ++i)
        invJacobian.mv(referenceGradients[i][0], gradients[i]);

      // compute matrix entries
      const auto weight = lf(quadPos);
      assert(not std::isnan(weight));

      const double z = weight * quad[pt].weight() * integrationElement;
      for (int i = 0; i < localView_.size(); ++i) {
        for (int j = i + 1; j < localView_.size(); ++j) {
          double zij = (gradients[i] * gradients[j]) * z;
          Dune::MatrixVector::addToDiagonal(localMatrix[i][j], zij);
          Dune::MatrixVector::addToDiagonal(localMatrix[j][i], zij);
        }
        Dune::MatrixVector::addToDiagonal(localMatrix[i][i],
                                          (gradients[i] * gradients[i]) *  z);
      }
    }
    return localMatrix;
  }

  void computeBulk()
  {
    const auto& fe = localView_.tree().finiteElement();
    insideCache_.bind(&fe);
    // using FE = std::decay_t<decltype(fe)>;
    // auto localAssembler = LaplaceAssembler<typename GV::Grid,
    // HPDG::AssemblyCache<FE>, HPDG::AssemblyCache<FE>>();
    auto quadKey =
      QuadratureRuleKey(localView_.element().type(), extraQuadOrder_);
    auto localAssembler =
      Dune::Fufem::DuneFunctionsWeightedMassAssembler<typename GV::Grid,
                                                      HPDG::AssemblyCache<FE>,
                                                      HPDG::AssemblyCache<FE>,
                                                      MassWeight>(
        basis_.gridView().grid(), massWeight_, quadKey);
    auto tmp = localMatrix_;
    localAssembler.assemble(
      localView_.element(), tmp, insideCache_, insideCache_);
    // localMatrix_+=tmp;

    for (std::size_t i = 0; i < fe.size(); i++) {
      auto row = localView_.index(i)[1];
      for (std::size_t j = 0; j < fe.size(); j++) {
        auto col = localView_.index(j)[1];
        localMatrix_[row][col] += factors_.mass * tmp[i][j];
      }
    }

    // now, we're left with the laplace part
    auto lap = laplaceBlock();
    for (std::size_t i = 0; i < fe.size(); i++) {
      auto row = localView_.index(i)[1];
      for (std::size_t j = 0; j < fe.size(); j++) {
        auto col = localView_.index(j)[1];
        localMatrix_[row][col] += factors_.laplace * lap[i][j];
      }
    }
  }

  // members:
  const Basis& basis_;
  const GradientWeight& gradWeight_;
  const MassWeight& massWeight_;
  int extraQuadOrder_;
  double penalty_;
  bool dirichlet_;
  LV localView_;
  LV outerView_;
  LocalMatrix localMatrix_;
  HPDG::AssemblyCache<FE> insideCache_;

  struct Factors {
    double mass = 1.0;
    double laplace = 1.0;
  };
  Factors factors_;
};
}
}
#endif
