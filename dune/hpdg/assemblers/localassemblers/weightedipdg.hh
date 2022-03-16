// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_WEIGHTED_IPDG_ASSEMBLER_HH
#define DUNE_HPDG_WEIGHTED_IPDG_ASSEMBLER_HH

// dune-common includes
#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>

// dune-istl includes
#include <dune/istl/matrix.hh>

// dune-fufem includes
#include "dune/fufem/assemblers/localassembler.hh"
#include "dune/fufem/quadraturerules/quadraturerulecache.hh"

// dune-matrix-vector includes
#include <dune/matrix-vector/addtodiagonal.hh>

namespace Dune::Fufem {

/** \brief Local assembler for edge contributions in the Interior Penalty Discontinuous Galerkin (IPDG) method
 *
 * This local assembler is meant to work in combination with the global IntersectionOperatorAssembler.
 * Note that mixed boundary conditions are not yet supported.
 *
 * For reference, see e.g.
 * > B. Riviere. Discontinuous Galerkin Methods for Solving Elliptic and Parabalic Equations. SIAM, 2008.
 */
template <class GridType, class F, class TrialLocalFE, class AnsatzLocalFE, class T=Dune::FieldMatrix<double,1,1>>
class WeightedInteriorPenaltyDGAssembler : public LocalAssembler<GridType, TrialLocalFE, AnsatzLocalFE,T>
{
    enum class DGType {SIPG = -1, IIPG = 0, NIPG = 1};

    using Base = LocalAssembler < GridType, TrialLocalFE, AnsatzLocalFE ,T >;
    static const int dim      = GridType::dimension -1; // We're on the edges
    static const int dimworld = GridType::dimensionworld;

  public:

    using Element = typename Base::Element;
    using BoolMatrix = typename Base::BoolMatrix;
    using LocalMatrix = typename Base::LocalMatrix;

    WeightedInteriorPenaltyDGAssembler(double penalty, const F& f, const QuadratureRuleKey& key, DGType dgType=DGType::SIPG) :
      sigma0_(penalty),
      f_(f),
      weightKey_(key),
      dgType_((double) dgType) {}

    /** \todo Please doc me! */
    void setPenalty(double penaltyParameter) {
      if(penaltyParameter<0.0)
        DUNE_THROW(Dune::MathError, "Penalty parameter must not be negative");
      sigma0_ = penaltyParameter;
    }

    void indices(const Element& element, BoolMatrix& isNonZero, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
      DUNE_THROW(Dune::Exception, "IPDG is called assembled edge-wise, not element-wise!");
    }

    template <class Edge>
    void indices(const Edge& it, BoolMatrix& isNonZero, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
      isNonZero = false; // For boundary edges, only in case of Dirichlet values we need to assemble a local matrix.
    }

    template <class Edge>
    void indices(const Edge& it, BoolMatrix& isNonZero, const TrialLocalFE& tFEi, const AnsatzLocalFE& aFEi, const TrialLocalFE& tFEo, const AnsatzLocalFE& aFEo) const
    {
      isNonZero = true;
    }


    /** Dummy function. We do not call assemble on elements here!
    */
    void assemble(const Element& element, LocalMatrix& localMatrix, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
      DUNE_THROW(Dune::Exception, "IPDG is called assembled edge-wise, not element-wise!");
    }

    /** \brief Assemble penalty term for boundary edges
     *
     * This method should be called for boundary edges, ie. those who only intersect with a single element.
     */
    template <class Edge>
    void assemble(const Edge& edge, LocalMatrix& localMatrix, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
      // For Neumann data, we don't assemble on boundary edges
      localMatrix = 0.0;
    }

    /** \brief Assemble penalty term for interior edges
     *
     * This method should be called for edges, ie. those who intersect two elements.
     *
     * (Note: 'edge' means a face with codim 1 here, so for 3d we mean a 2d face (usually a triangle).)
     *
     */
    template <class Edge, class MC>
    void assembleBlockwise(const Edge& edge, MC& matrixContainer, const TrialLocalFE& tFEinside, const AnsatzLocalFE& aFEinside, const TrialLocalFE& tFEoutside, const AnsatzLocalFE& aFEoutside) const
    {

      const auto tP = tFEinside.localBasis().order();
      const auto aP = aFEoutside.localBasis().order();
      const auto maxOrder = std::max(tP, aP);
      const auto penalty = sigma0_ * maxOrder*maxOrder; // TODO: ^2 scaling is wrong for dim==1

      using FVdimworld = typename Dune::template FieldVector<double,dimworld>;

      using JacobianType = typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType;
      using RangeType = typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType;

      // get geometry and store it
      const auto edgeGeometry = edge.geometry();
      const auto insideGeometry = edge.inside().geometry();
      const auto outsideGeometry = edge.outside().geometry();

      const auto edgeLength = edgeGeometry.volume();
      matrixContainer = 0.0;

      // get quadrature rule
      QuadratureRuleKey tFEquad(edge.type(), std::max(tFEinside.localBasis().order(),tFEoutside.localBasis().order()));
      QuadratureRuleKey quadKey = tFEquad.square();

      const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey.product(weightKey_));

      // store gradients for both the inner and outer elements
      const int innerSize = tFEinside.localBasis().size();
      const int outerSize = tFEoutside.localBasis().size();
      std::vector<JacobianType> insideReferenceGradients(innerSize);
      std::vector<JacobianType> outsideReferenceGradients(outerSize);
      std::vector<FVdimworld> insideGradients(innerSize);
      std::vector<FVdimworld> outsideGradients(outerSize);

      // store values of shape functions
      std::vector<RangeType> tFEinsideValues(innerSize);
      std::vector<RangeType> tFEoutsideValues(outerSize);

      const auto outerNormal = edge.centerUnitOuterNormal();
      auto inner_lf = localFunction(f_);
      inner_lf.bind(edge.inside());
      auto outer_lf = localFunction(f_);
      outer_lf.bind(edge.outside());

      // loop over quadrature points
      for (size_t pt=0; pt < quad.size(); ++pt)
      {
        // get quadrature point
        const auto& quadPos = quad[pt].position();
        const auto gf_inside_val = inner_lf(edge.geometryInInside().global(quadPos));
        const auto gf_outside_val = outer_lf(edge.geometryInOutside().global(quadPos));
        // Now that I have the edge values, I also want to average.
        // C++ 20: std::midpoint
        const auto weight_avg = 0.5 * (gf_inside_val + gf_outside_val);

        // get transposed inverse of Jacobian of transformation
        const auto& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));
        const auto& outsideInvJacobian = outsideGeometry.jacobianInverseTransposed(edge.geometryInOutside().global(quadPos));

        // get integration factor
        const auto integrationElement = edgeGeometry.integrationElement(quadPos);

        // get gradients of shape functions on both the inside and outside element
        tFEinside.localBasis().evaluateJacobian(edge.geometryInInside().global(quadPos), insideReferenceGradients);
        tFEoutside.localBasis().evaluateJacobian(edge.geometryInOutside().global(quadPos), outsideReferenceGradients);

        // transform gradients
        for (size_t i=0; i<innerSize; ++i) {
          invJacobian.mv(insideReferenceGradients[i][0], insideGradients[i]);
          insideGradients[i] *= gf_inside_val;
        }
        for (size_t i=0; i<outerSize; ++i) {
          outsideInvJacobian.mv(outsideReferenceGradients[i][0], outsideGradients[i]);
          outsideGradients[i] *= gf_outside_val;
        }

        // evaluate basis functions
        tFEinside.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos), tFEinsideValues);
        tFEoutside.localBasis().evaluateFunction(edge.geometryInOutside().global(quadPos), tFEoutsideValues);

        // compute matrix entries
        const auto z = quad[pt].weight() * integrationElement;

        // Basis functions from inside as test functions
        for (size_t i=0; i<innerSize; ++i)
        {
          // Basis functions from inside as ansatz functions
          for (size_t j=0; j<innerSize; ++j)
          {
            // M11, see Riviere, p. 54f
            auto zij = -0.5*z*tFEinsideValues[i]*(insideGradients[j]*outerNormal);
            zij += 0.5*dgType_*z*tFEinsideValues[j]*(insideGradients[i]*outerNormal);
            zij += penalty * weight_avg *z/edgeLength*tFEinsideValues[i]*tFEinsideValues[j];
            Dune::MatrixVector::addToDiagonal(matrixContainer[0][0][i][j],zij);
          }
          // Basis functions from outside as ansatz functions
          for (size_t j=0; j<outerSize; ++j) {
            // M12
            auto zij = -0.5*z*tFEinsideValues[i]*(outsideGradients[j]*outerNormal);
            zij -= 0.5*dgType_*z*tFEoutsideValues[j]*(insideGradients[i]*outerNormal);
            zij -= penalty*weight_avg*z/edgeLength*tFEinsideValues[i]*tFEoutsideValues[j];
            Dune::MatrixVector::addToDiagonal(matrixContainer[0][1][i][j],zij);
          }
        }
        // Basis functions from outside as test functions
        for (size_t i=0; i<outerSize; ++i) {
          // Basis functions from inside as ansatz functions
          for (size_t j=0; j<innerSize; ++j) {
            // M21, see Riviere, p. 54f
            auto zij = 0.5*z*tFEoutsideValues[i]*(insideGradients[j]*outerNormal);
            zij += 0.5*dgType_*z*tFEinsideValues[j]*(outsideGradients[i]*outerNormal);
            zij -= penalty*weight_avg*z/edgeLength*tFEoutsideValues[i]*tFEinsideValues[j];
            Dune::MatrixVector::addToDiagonal(matrixContainer[1][0][i][j],zij);
          }
          // Basis functions from outside as ansatz functions
          for (size_t j=0; j<outerSize; ++j) {
            // M22
            auto zij = 0.5*z*tFEoutsideValues[i]*(outsideGradients[j]*outerNormal);
            zij -= 0.5*dgType_*z*tFEoutsideValues[j]*(outsideGradients[i]*outerNormal);
            zij += penalty*weight_avg*z/edgeLength*tFEoutsideValues[i]*tFEoutsideValues[j];
            Dune::MatrixVector::addToDiagonal(matrixContainer[1][1][i][j],zij);
          }
        }
      }
    }
  private:


    /** Penalty term \sigma_0
     *
     * This term penalizes discontinuities across the edges.
     * Should be chosed large enough. In particular, it should scale approx. in k^2 where k
     * is the polynomial degree of the FE space.
     */
    double sigma0_; // Penalizes discontinuities

    F f_;
    QuadratureRuleKey weightKey_;

    const double dgType_; // -1 leads to the symmetric interior penalty DG variant. {-1,0,1} are possible values.

};
}
#endif
