// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef V_IPDG_ASSEMBLER_HH
#define V_IPDG_ASSEMBLER_HH

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


/** \brief Local assembler for edge contributions in the Interior Penalty Discontinuous Galerkin (IPDG) method
 *
 * This local assembler is meant to work in combination with the global IntersectionOperatorAssembler.
 * Note that mixed boundary conditions are not yet supported.
 *
 * For reference, see e.g.
 * > B. Riviere. Discontinuous Galerkin Methods for Solving Elliptic and Parabalic Equations. SIAM, 2008.
 */
template <class GridType, class TrialLocalFE, class AnsatzLocalFE, class T=Dune::FieldMatrix<double,1,1>>
class VInteriorPenaltyDGAssembler : public LocalAssembler<GridType, TrialLocalFE, AnsatzLocalFE,T>
{
    enum class DGType {SIPG = -1, IIPG = 0, NIPG = 1};

    using Base = LocalAssembler < GridType, TrialLocalFE, AnsatzLocalFE ,T >;
    static const int dim      = GridType::dimension -1; // We're on the edges
    static const int dimworld = GridType::dimensionworld;

  public:

    using Element = typename Base::Element;
    using BoolMatrix = typename Base::BoolMatrix;
    using LocalMatrix = typename Base::LocalMatrix;

    VInteriorPenaltyDGAssembler() :
      sigma0_(0),
      dirichlet_(true),
      sigma1_(0),
      dgType_((double) DGType::SIPG) {}

    VInteriorPenaltyDGAssembler(double penalty, bool dirichlet=true, DGType dgType=DGType::SIPG, double gradientPenalty=0) :
      sigma0_(penalty),
      dirichlet_(dirichlet),
      dgType_((double) dgType),
      sigma1_(gradientPenalty) {}

    /** \todo Please doc me! */
    void setPenalty(double penaltyParameter) {
      if(penaltyParameter<0.0)
        DUNE_THROW(Dune::MathError, "Penalty parameter must not be negative");
      sigma0_ = penaltyParameter;
    }

    /** \todo Please doc me! */
    void setGradientPenalty(double gradientPenalty) {
      if(gradientPenalty<0.0)
        DUNE_THROW(Dune::MathError, "Penalty parameter must not be negative");
      sigma1_ = gradientPenalty;
    }

    void indices(const Element& element, BoolMatrix& isNonZero, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
      DUNE_THROW(Dune::Exception, "IPDG is called assembled edge-wise, not element-wise!");
    }

    template <class Edge>
    void indices(const Edge& it, BoolMatrix& isNonZero, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
      isNonZero = dirichlet_; // For boundary edges, only in case of Dirichlet values we need to assemble a local matrix.
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

      localMatrix = 0.0;

      // For Neumann data, we don't assemble on boundary edges
      // TODO: There is no need that this has to be decided globally. It would be very easy to
      // just decide this edge by edge for mixed b.c..
      if (!dirichlet_)
        return;

      auto tP = tFE.localBasis().order();
      auto aP = aFE.localBasis().order();
      auto maxOrder = std::max(tP, aP);
      auto penalty = sigma0_ * maxOrder*maxOrder; // TODO: ^2 scaling is wrong for dim==1

      using FVdimworld = typename Dune::template FieldVector<double,dimworld>;

      using JacobianType = typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType;
      using RangeType = typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType;

      auto rows = localMatrix.N();
      auto cols = localMatrix.M();

      // get geometries of the edge and the inside element
      const auto edgeGeometry = edge.geometry();
      const auto insideGeometry = edge.inside().geometry();

      // Get the length of the edge
      const auto edgeLength= edgeGeometry.volume();

      // get quadrature rule
      QuadratureRuleKey tFEquad(edge.type(), tFE.localBasis().order());
      QuadratureRuleKey quadKey = tFEquad.square();

      const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

      // store gradients of shape functions and base functions
      std::vector<JacobianType> referenceGradients(tFE.localBasis().size());
      std::vector<FVdimworld> gradients(tFE.localBasis().size());

      // store values of shape functions
      std::vector<RangeType> tFEvalues(tFE.localBasis().size());

      // We also need a unit normal on the edge pointing away from the inner element
      const auto outerNormal = edge.centerUnitOuterNormal();

      // loop over quadrature points
      for (size_t pt=0; pt < quad.size(); ++pt)
      {
        // get quadrature point
        const auto& quadPos = quad[pt].position();

        // get transposed inverse of Jacobian of transformation
        const auto& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));

        // get integration factor
        const auto integrationElement = edgeGeometry.integrationElement(quadPos);

        // get gradients of shape functions
        tFE.localBasis().evaluateJacobian(edge.geometryInInside().global(quadPos), referenceGradients);

        // transform gradients
        for (size_t i=0; i<gradients.size(); ++i)
          invJacobian.mv(referenceGradients[i][0], gradients[i]);

        // evaluate basis functions
        tFE.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos), tFEvalues);

        // compute matrix entries
        auto z = quad[pt].weight() * integrationElement;
        for (size_t i=0; i<rows; ++i)
        {
          for (size_t j=0; j<cols; ++j)
          {
            auto zij = -z*tFEvalues[i]*(gradients[j]*outerNormal);
            zij += dgType_*z*tFEvalues[j]*(gradients[i]*outerNormal);
            zij += penalty*z/edgeLength*tFEvalues[i]*tFEvalues[j];
            // The sigma1_ stabilization term does not effect boundary edges
            Dune::MatrixVector::addToDiagonal(localMatrix[i][j],zij);
          }
        }
      }
    }

    /** \brief Assemble penalty term for interior edges
     *
     * This method should be called for edges, ie. those who intersect two elements.
     *
     * (Note: 'edge' means a face with codim 1 here, so for 3d we mean a 2d face (usually a triangle).)
     *
     * \warning The localMatrix will be organized as in the classis IntersectionOperatorAssembler.
     */
    template <class Edge>
    void assemble(const Edge& edge, LocalMatrix& localMatrix, const TrialLocalFE& tFEinside, const AnsatzLocalFE& aFEinside, const TrialLocalFE& tFEoutside, const AnsatzLocalFE& aFEoutside) const
    {

      localMatrix = 0.0;

      // to avoid too much duplicate code, we write this method, which was designed for the vintage intersection operator assembler, by calling
      // the newer assembleBlockwise method.
      using MatrixContainer = Dune::Matrix<LocalMatrix>;
      auto mc = MatrixContainer(2,2);
      mc[0][0].setSize(tFEinside.size(), aFEinside.size());
      mc[0][1].setSize(tFEinside.size(), aFEoutside.size());
      mc[1][0].setSize(tFEoutside.size(), aFEinside.size());
      mc[1][1].setSize(tFEoutside.size(), aFEoutside.size());

      // compute entries
      assembleBlockwise(edge, mc, tFEinside, aFEinside, tFEoutside, aFEoutside);

      // redistribute them into the localMatrix
      auto rows = localMatrix.N();
      auto cols = localMatrix.M();

      const auto insideSize = tFEinside.localBasis().size();

      for (size_t i=0; i<rows; ++i)
      {
        for (size_t j=0; j<cols; ++j)
        {
          // Depending on which elements i and j belong to, perform different operations.
          // In particular, i (resp. j) < insizeSize corresponds to functions from the inner element
          // while the others correspond to the outer element
          if (i < insideSize){
            if(j< insideSize)
              Dune::MatrixVector::addToDiagonal(localMatrix[i][j], mc[0][0][i][j]);
            else
              Dune::MatrixVector::addToDiagonal(localMatrix[i][j], mc[0][1][i][j-insideSize]);
          }
          else {
            if(j< insideSize)
              Dune::MatrixVector::addToDiagonal(localMatrix[i][j], mc[1][0][i-insideSize][j]);
            else
              Dune::MatrixVector::addToDiagonal(localMatrix[i][j], mc[1][1][i-insideSize][j-insideSize]);
          }
        }
      }
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

      auto tP = tFEinside.localBasis().order();
      auto aP = aFEoutside.localBasis().order();
      auto maxOrder = std::max(tP, aP);
      auto penalty = sigma0_ * maxOrder*maxOrder; // TODO: ^2 scaling is wrong for dim==1

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

      const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

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

      // loop over quadrature points
      for (size_t pt=0; pt < quad.size(); ++pt)
      {
        // get quadrature point
        const auto& quadPos = quad[pt].position();

        // get transposed inverse of Jacobian of transformation
        const auto& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));
        const auto& outsideInvJacobian = outsideGeometry.jacobianInverseTransposed(edge.geometryInOutside().global(quadPos));

        // get integration factor
        const auto integrationElement = edgeGeometry.integrationElement(quadPos);

        // get gradients of shape functions on both the inside and outside element
        tFEinside.localBasis().evaluateJacobian(edge.geometryInInside().global(quadPos), insideReferenceGradients);
        tFEoutside.localBasis().evaluateJacobian(edge.geometryInOutside().global(quadPos), outsideReferenceGradients);

        // transform gradients
        for (size_t i=0; i<innerSize; ++i)
          invJacobian.mv(insideReferenceGradients[i][0], insideGradients[i]);
        for (size_t i=0; i<outerSize; ++i)
          outsideInvJacobian.mv(outsideReferenceGradients[i][0], outsideGradients[i]);

        // evaluate basis functions
        tFEinside.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos), tFEinsideValues);
        tFEoutside.localBasis().evaluateFunction(edge.geometryInOutside().global(quadPos), tFEoutsideValues);

        // compute matrix entries
        auto z = quad[pt].weight() * integrationElement;

        // Basis functions from inside as test functions
        for (size_t i=0; i<innerSize; ++i)
        {
          // Basis functions from inside as ansatz functions
          for (size_t j=0; j<innerSize; ++j)
          {
            // M11, see Riviere, p. 54f
            auto zij = -0.5*z*tFEinsideValues[i]*(insideGradients[j]*outerNormal);
            zij += 0.5*dgType_*z*tFEinsideValues[j]*(insideGradients[i]*outerNormal);
            zij += penalty*z/edgeLength*tFEinsideValues[i]*tFEinsideValues[j];
            zij += sigma1_*z/edgeLength*(insideGradients[i]*outerNormal)*(insideGradients[j]*outerNormal);
            Dune::MatrixVector::addToDiagonal(matrixContainer[0][0][i][j],zij);
          }
          // Basis functions from outside as ansatz functions
          for (size_t j=0; j<outerSize; ++j) {
            // M12
            auto zij = -0.5*z*tFEinsideValues[i]*(outsideGradients[j]*outerNormal);
            zij -= 0.5*dgType_*z*tFEoutsideValues[j]*(insideGradients[i]*outerNormal);
            zij -= penalty*z/edgeLength*tFEinsideValues[i]*tFEoutsideValues[j];
            zij -= sigma1_*z/edgeLength*(insideGradients[i]*outerNormal)*(outsideGradients[j]*outerNormal);
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
            zij -= penalty*z/edgeLength*tFEoutsideValues[i]*tFEinsideValues[j];
            zij -= sigma1_*z/edgeLength*(outsideGradients[i]*outerNormal)*(insideGradients[j]*outerNormal);
            Dune::MatrixVector::addToDiagonal(matrixContainer[1][0][i][j],zij);
          }
          // Basis functions from outside as ansatz functions
          for (size_t j=0; j<outerSize; ++j) {
            // M22
            auto zij = 0.5*z*tFEoutsideValues[i]*(outsideGradients[j]*outerNormal);
            zij -= 0.5*dgType_*z*tFEoutsideValues[j]*(outsideGradients[i]*outerNormal);
            zij += penalty*z/edgeLength*tFEoutsideValues[i]*tFEoutsideValues[j];
            zij += sigma1_*z/edgeLength*(outsideGradients[i]*outerNormal)*(outsideGradients[j]*outerNormal);
            Dune::MatrixVector::addToDiagonal(matrixContainer[1][1][i][j],zij);
          }
        }
      }
    }

    template <class Edge, class MC>
    void assembleBlockwisePenalty(const Edge& edge, MC& matrixContainer, const TrialLocalFE& tFEinside, const AnsatzLocalFE& aFEinside, const TrialLocalFE& tFEoutside, const AnsatzLocalFE& aFEoutside) const
    {

      auto tP = tFEinside.localBasis().order();
      auto aP = aFEoutside.localBasis().order();
      auto maxOrder = std::max(tP, aP);
      auto penalty = sigma0_ * maxOrder*maxOrder; // TODO: ^2 scaling is wrong for dim==1

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
      QuadratureRuleKey quadKey = tFEquad.derivative().square();

      const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

      // store gradients for both the inner and outer elements
      std::vector<JacobianType> insideReferenceGradients(tFEinside.localBasis().size());
      std::vector<JacobianType> outsideReferenceGradients(tFEoutside.localBasis().size());
      std::vector<FVdimworld> insideGradients(tFEinside.localBasis().size());
      std::vector<FVdimworld> outsideGradients(tFEoutside.localBasis().size());

      // store values of shape functions
      std::vector<RangeType> tFEinsideValues(tFEinside.localBasis().size());
      std::vector<RangeType> tFEoutsideValues(tFEoutside.localBasis().size());

      const auto outerNormal = edge.centerUnitOuterNormal();

      // loop over quadrature points
      for (size_t pt=0; pt < quad.size(); ++pt)
      {
        // get quadrature point
        const auto& quadPos = quad[pt].position();

        // get transposed inverse of Jacobian of transformation
        const auto& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));
        const auto& outsideInvJacobian = outsideGeometry.jacobianInverseTransposed(edge.geometryInOutside().global(quadPos));

        // get integration factor
        const auto integrationElement = edgeGeometry.integrationElement(quadPos);

        // get gradients of shape functions on both the inside and outside element
        tFEinside.localBasis().evaluateJacobian(edge.geometryInInside().global(quadPos), insideReferenceGradients);
        tFEoutside.localBasis().evaluateJacobian(edge.geometryInOutside().global(quadPos), outsideReferenceGradients);

        // transform gradients
        for (size_t i=0; i<insideGradients.size(); ++i)
          invJacobian.mv(insideReferenceGradients[i][0], insideGradients[i]);
        for (size_t i=0; i<outsideGradients.size(); ++i)
          outsideInvJacobian.mv(outsideReferenceGradients[i][0], outsideGradients[i]);

        // evaluate basis functions
        tFEinside.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos), tFEinsideValues);
        tFEoutside.localBasis().evaluateFunction(edge.geometryInOutside().global(quadPos), tFEoutsideValues);

        // compute matrix entries
        auto z = quad[pt].weight() * integrationElement;

        // Basis functions from inside as test functions
        for (size_t i=0; i<tFEinside.localBasis().size(); ++i)
        {
          // Basis functions from inside as ansatz functions
          for (size_t j=0; j<aFEinside.localBasis().size(); ++j)
          {
            // M11, see Riviere, p. 54f
            auto zij = penalty*z/edgeLength*tFEinsideValues[i]*tFEinsideValues[j];
            Dune::MatrixVector::addToDiagonal(matrixContainer[0][0][i][j],zij);
          }
          // Basis functions from outside as ansatz functions
          for (size_t j=0; j<aFEoutside.localBasis().size(); ++j) {
            // M12
            auto zij = -penalty*z/edgeLength*tFEinsideValues[i]*tFEoutsideValues[j];
            Dune::MatrixVector::addToDiagonal(matrixContainer[0][1][i][j],zij);
          }
        }
        // Basis functions from outside as test functions
        for (size_t i=0; i<tFEoutside.localBasis().size(); ++i) {
          // Basis functions from inside as ansatz functions
          for (size_t j=0; j<aFEinside.localBasis().size(); ++j) {
            // M21, see Riviere, p. 54f
            auto zij = -penalty*z/edgeLength*tFEoutsideValues[i]*tFEinsideValues[j];
            Dune::MatrixVector::addToDiagonal(matrixContainer[1][0][i][j],zij);
          }
          // Basis functions from outside as ansatz functions
          for (size_t j=0; j<aFEoutside.localBasis().size(); ++j) {
            // M22
            auto zij = penalty*z/edgeLength*tFEoutsideValues[i]*tFEoutsideValues[j];
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

    /** Penalty term \sigma_1
     *
     * This term penalizes jumps in the derivatives across the edges.
     */

    /** Flag that checks if dirichlet_ terms must be assembled for the edges or not.*/
    bool dirichlet_;

    const double dgType_; // -1 leads to the symmetric interior penalty DG variant. {-1,0,1} are possible values.

    double sigma1_; // Penalizes jumps of the normal derivative along the non-boundary edges
};
#endif
