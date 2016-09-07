#ifndef IPDG_ASSEMBLER_HH
#define IPDG_ASSEMBLER_HH

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
 *
 * For reference, see e.g.
 * > B. Riviere. Discontinuous Galerkin Methods for Solving Elliptic and Parabalic Equations. SIAM, 2008.
 */
template <class GridType, class TrialLocalFE, class AnsatzLocalFE, class T=Dune::FieldMatrix<double,1,1>>
class InteriorPenaltyDGAssembler : public LocalAssembler<GridType, TrialLocalFE, AnsatzLocalFE,T>
{
private:
    typedef LocalAssembler < GridType, TrialLocalFE, AnsatzLocalFE ,T > Base;
    static const int dim      = GridType::dimension -1; // We're on the edges
    static const int dimworld = GridType::dimensionworld;
    const int quadOrder_ =-1;

    // TODO: Maybe use enum for the IP types
    const double DGType = -1.0; // -1 leads to the symmetric interior penalty DG variant. {-1,0,1} are possible values.

public:

    // TODO sigma_e^0 should be carefully chosen (and should probably be private)
    bool dirichlet = true;
    double sigma0 = 50.0; // Penalizes discontinuities
    double sigma1 = 0; // Penalizes jumps of the normal derivative along the non-boundary edges

    typedef typename Base::Element Element;
    typedef typename Element::Geometry Geometry;
    typedef typename Geometry::JacobianInverseTransposed JacobianInverseTransposed;
    typedef typename Base::BoolMatrix BoolMatrix;
    typedef typename Base::LocalMatrix LocalMatrix;

    void indices(const Element& element, BoolMatrix& isNonZero, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
//        isNonZero = true;
        DUNE_THROW(Dune::NotImplemented, "IPDG is called assembled edge-wise, not element-wise!");
    }

    template <class Edge>
    void indices(const Edge& it, BoolMatrix& isNonZero, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
        isNonZero = dirichlet; //TODO
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
        DUNE_THROW(Dune::NotImplemented, "IPDG is called assembled edge-wise, not element-wise!");
    }

    /** \brief Assemble penalty term for boundary edges
     *
     * This method should be called for boundary edges, ie. those who only intersect with a single element.
     */
    template <class Edge>
    void assemble(const Edge& edge, LocalMatrix& localMatrix, const TrialLocalFE& tFE, const AnsatzLocalFE& aFE) const
    {
        typedef typename Dune::template FieldVector<double,dim> FVdim;
        typedef typename Dune::template FieldVector<double,dimworld> FVdimworld;

        typedef typename Edge::Geometry Geometry;
        typedef typename Edge::Entity Entity;
        typedef typename Entity::Geometry InsideGeometry;
        typedef typename InsideGeometry::JacobianInverseTransposed JacobianInverseTransposed;
        typedef typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType JacobianType;
        typedef typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType RangeType;

        // check if ansatz local fe = test local fe
        if (not Base::isSameFE(tFE, aFE))
            DUNE_THROW(Dune::NotImplemented, "IPDG is only implemented for ansatz space=test space!");

        int rows = localMatrix.N();
        int cols = localMatrix.M();

        localMatrix = 0.0;
        // For Neumann data, we don't assemble on boundary edges
        if (!dirichlet)
            return;

        // get geometries of the edge and the inside element
        const Geometry edgeGeometry = edge.geometry();
        const InsideGeometry insideGeometry = edge.inside().geometry();

        // Get the length of the edge
        const double& edgeLength= edgeGeometry.volume();


        // get quadrature rule
        QuadratureRuleKey tFEquad(edge.type(), tFE.localBasis().order());
        QuadratureRuleKey quadKey = tFEquad.derivative().square();

        const Dune::template QuadratureRule<double, dim>& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

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
            const FVdim& quadPos = quad[pt].position();

            // get transposed inverse of Jacobian of transformation
            const JacobianInverseTransposed& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));

            // get integration factor
            const double integrationElement = edgeGeometry.integrationElement(quadPos);

            // get gradients of shape functions
            tFE.localBasis().evaluateJacobian(edge.geometryInInside().global(quadPos), referenceGradients);

            // transform gradients
            for (size_t i=0; i<gradients.size(); ++i)
                invJacobian.mv(referenceGradients[i][0], gradients[i]);

            // evaluate basis functions
            tFE.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos), tFEvalues);

            // compute matrix entries
            double z = quad[pt].weight() * integrationElement;
            for (size_t i=0; i<rows; ++i)
            {
                for (size_t j=0; j<cols; ++j)
                {
                    double zij = -0.5*z*tFEvalues[i]*(gradients[j]*outerNormal);
                    zij += 0.5*DGType*z*tFEvalues[j]*(gradients[i]*outerNormal);
                    // TODO: This should be multiplied by k^2 (where k is the polynomial degree of the basis functions)
                    zij += sigma0*z/edgeLength*tFEvalues[i]*tFEvalues[j];
                    // The sigma1 stabilization term does not effect boundary edges
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
     */
    template <class Edge>
    void assemble(const Edge& edge, LocalMatrix& localMatrix, const TrialLocalFE& tFEinside, const AnsatzLocalFE& aFEinside, const TrialLocalFE& tFEoutside, const AnsatzLocalFE& aFEoutside) const
    {
        typedef typename Dune::template FieldVector<double,dim> FVdim;
        typedef typename Dune::template FieldVector<double,dimworld> FVdimworld;

        typedef typename Edge::Geometry Geometry;
        typedef typename Edge::Entity Entity;
        typedef typename Entity::Geometry InsideGeometry;
        typedef typename InsideGeometry::JacobianInverseTransposed JacobianInverseTransposed;
        typedef typename InsideGeometry::JacobianInverseTransposed JacobianInverseTransposed;
        typedef typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType JacobianType;
        typedef typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType RangeType;

        // check if ansatz local fe = test local fe
        if (not (Base::isSameFE(tFEinside, aFEinside) and Base::isSameFE(tFEoutside, aFEoutside)))
            DUNE_THROW(Dune::NotImplemented, "IPDG is only implemented for ansatz space=test space!");

        int rows = localMatrix.N();
        int cols = localMatrix.M();

        // get geometry and store it
        const Geometry edgeGeometry = edge.geometry(); //TODO: does this work for edges?
        const InsideGeometry insideGeometry = edge.inside().geometry();
        const auto outsideGeometry = edge.outside().geometry();

        const double& edgeLength = edgeGeometry.volume(); // TODO: Is this what I want?
        localMatrix = 0.0;

        // get quadrature rule
        QuadratureRuleKey tFEquad(edge.type(), tFEinside.localBasis().order());
        QuadratureRuleKey quadKey = tFEquad.derivative().square();

        const Dune::template QuadratureRule<double, dim>& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

        // store gradients of shape functions and base functions
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
            const FVdim& quadPos = quad[pt].position();

            // get transposed inverse of Jacobian of transformation
            const JacobianInverseTransposed& invJacobian = insideGeometry.jacobianInverseTransposed(edge.geometryInInside().global(quadPos));
            const auto& outsideInvJacobian = outsideGeometry.jacobianInverseTransposed(edge.geometryInOutside().global(quadPos));

            // get integration factor
            const double integrationElement = edgeGeometry.integrationElement(quadPos);

            // get gradients of shape functions on both the inside and outside element
            tFEinside.localBasis().evaluateJacobian(edge.geometryInInside().global(quadPos), insideReferenceGradients);
            tFEoutside.localBasis().evaluateJacobian(edge.geometryInOutside().global(quadPos), outsideReferenceGradients);

            // transform gradients
            for (size_t i=0; i<insideGradients.size(); ++i)
                invJacobian.mv(insideReferenceGradients[i][0], insideGradients[i]);
            for (size_t i=0; i<insideGradients.size(); ++i)
                outsideInvJacobian.mv(outsideReferenceGradients[i][0], outsideGradients[i]);

            // evaluate basis functions
            tFEinside.localBasis().evaluateFunction(edge.geometryInInside().global(quadPos), tFEinsideValues);
            tFEoutside.localBasis().evaluateFunction(edge.geometryInOutside().global(quadPos), tFEoutsideValues);

            // compute matrix entries
            double z = quad[pt].weight() * integrationElement;

            const size_t& insideSize = tFEinside.localBasis().size();

            for (size_t i=0; i<rows; ++i)
            {
                for (size_t j=0; j<cols; ++j)
                {
                    // Depending on which elements i and j belong to, perform different operations.
                    // In particular, i (resp. j) < insizeSize corresponds to functions from the inner element
                    // while the others correspond to the outer element
                    if (i < insideSize){
                        if(j< insideSize) {
                            // M11, see Riviere, p. 54f
                            double zij = -0.5*z*tFEinsideValues[i]*(insideGradients[j]*outerNormal);
                            zij += 0.5*DGType*z*tFEinsideValues[j]*(insideGradients[i]*outerNormal);
                            zij += sigma0*z/edgeLength*tFEinsideValues[i]*tFEinsideValues[j];
                            zij += sigma1*z/edgeLength*(insideGradients[i]*outerNormal)*(insideGradients[j]*outerNormal);
                            Dune::MatrixVector::addToDiagonal(localMatrix[i][j],zij);
                        }
                        else {
                            const size_t jj = j-insideSize;
                            // M12
                            double zij = -0.5*z*tFEinsideValues[i]*(outsideGradients[jj]*outerNormal);
                            zij -= 0.5*DGType*z*tFEoutsideValues[jj]*(insideGradients[i]*outerNormal);
                            zij -= sigma0*z/edgeLength*tFEinsideValues[i]*tFEoutsideValues[jj];
                            zij -= sigma1*z/edgeLength*(insideGradients[i]*outerNormal)*(outsideGradients[jj]*outerNormal);
                            Dune::MatrixVector::addToDiagonal(localMatrix[i][j],zij);
                        }
                    }
                    else {
                        const size_t ii = i-insideSize;
                        if(j< insideSize) {
                            // M21, see Riviere, p. 54f
                            double zij = 0.5*z*tFEoutsideValues[ii]*(insideGradients[j]*outerNormal);
                            zij += 0.5*DGType*z*tFEinsideValues[j]*(outsideGradients[ii]*outerNormal);
                            zij -= sigma0*z/edgeLength*tFEoutsideValues[ii]*tFEinsideValues[j];
                            zij -= sigma1*z/edgeLength*(outsideGradients[ii]*outerNormal)*(insideGradients[j]*outerNormal);
                            Dune::MatrixVector::addToDiagonal(localMatrix[i][j],zij);
                        }
                        else {
                            // M22
                            const size_t jj = j-insideSize;
                            double zij = 0.5*z*tFEoutsideValues[ii]*(outsideGradients[jj]*outerNormal);
                            zij -= 0.5*DGType*z*tFEoutsideValues[jj]*(outsideGradients[ii]*outerNormal);
                            zij += sigma0*z/edgeLength*tFEoutsideValues[ii]*tFEoutsideValues[jj];
                            zij += sigma1*z/edgeLength*(outsideGradients[ii]*outerNormal)*(outsideGradients[jj]*outerNormal);
                            Dune::MatrixVector::addToDiagonal(localMatrix[i][j],zij);
                        }

                    }
                }
            }
        }
    }
};
#endif
