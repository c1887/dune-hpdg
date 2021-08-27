// -*- tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set ts=8 sw=4 et sts=4:
#ifndef IPDG_BOUNDARY_ASSEMBLER_HH
#define IPDG_BOUNDARY_ASSEMBLER_HH

#include <memory>

#include <dune/common/fvector.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/istl/bvector.hh>

#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

#include <dune/fufem/assemblers/localboundaryassembler.hh>

namespace Dune::HPDG {

/** \brief Assemble right hand side contributions of the IPDG method
 * 
 * \tparam GridType The grid we are assembling for
 * \tparam T Type used for the set of dofs at one node
 */
template <class GridType, class Function, class T=Dune::FieldVector<typename GridType::ctype,1> >
class IPDGBoundaryAssembler :
    public LocalBoundaryAssembler<GridType, T>

    /** TODO: The quadrature order is currently fixed such that
     * the test functions are square integrable. This might be
     * too few or too much depending on the user supplied functional!
     */

{
    private:
        static const int dim = GridType::dimension;
        using ctype = typename GridType::ctype;
        static const int dimworld = GridType::dimensionworld;

        using LocalFunction = std::decay_t<decltype(localFunction(std::declval<Function>()))>;

    public:
        using LocalVector = typename LocalBoundaryAssembler<GridType,T>::LocalVector;


        /** \brief Constructor
         * \param f Neumann force function or Dirichlet data
         * \param bdrType bool: true is for Dirichlet, false for Neumann
         * \param varyingDegree bool: set true if degrees on elements may be different
         */
        template <class FF>
        IPDGBoundaryAssembler(FF&& f, bool bdrType=true, bool varyingDegree=false) :
            bdrFunction_(Dune::wrap_or_move(f)),
            lf_(localFunction(*bdrFunction_)),
            dirichlet(bdrType),
            varyingDegree_(varyingDegree)
        {}


        // TODO:
        template <class TrialLocalFE, class BoundaryIterator>
        void assemble(const BoundaryIterator& it, LocalVector& localVector, const TrialLocalFE& tFE)
        {
            if (dirichlet)
                assembleDirichlet(it, localVector, tFE);
            else
                assembleNeumann(it, localVector, tFE);
        }

        template <class TrialLocalFE, class BoundaryIterator>
        void assembleDirichlet(const BoundaryIterator& it, LocalVector& localVector, const TrialLocalFE& tFE)
        {
            using RangeType = typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType;

            localVector = 0.0;

            double penalty = penalty_;
            if (varyingDegree_) {
                auto degree = tFE.localBasis().order();
                penalty*= degree*degree;
            }
            const auto edgeLength = it->geometry().volume();
            penalty/=edgeLength;


            // geometry of the boundary face
            const auto& segmentGeometry = it->geometry();

            // get quadrature rule
            QuadratureRuleKey tFEquad(it->type(), tFE.localBasis().order());
            auto quadKey = tFEquad.square();

            const auto& quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

            // store values of shape functions
            std::vector<RangeType> values(tFE.localBasis().size());
            std::vector<typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType> refGradients(tFE.localBasis().size());
            std::vector<Dune::FieldVector<ctype, dimworld>> gradients(tFE.localBasis().size());

            const auto& inside = it->inside();

            lf_.bind(inside);

            const auto outerNormal = it->centerUnitOuterNormal();


            // loop over quadrature points
            for (size_t pt=0; pt < quad.size(); ++pt)
            {

                // get quadrature point
                const auto& quadPos = quad[pt].position();

                // get integration factor
                const auto integrationElement = segmentGeometry.integrationElement(quadPos);

                // position of the quadrature point within the element
                const auto& elementQuadPos = it->geometryInInside().global(quadPos);

                // get transposed inverse of Jacobian of transformation
                const auto& invJacobian = inside.geometry().jacobianInverseTransposed(elementQuadPos);

                // Evaluate gradients
                tFE.localBasis().evaluateJacobian(elementQuadPos, refGradients);
                for (size_t i =0; i< gradients.size(); i++)
                    invJacobian.mv(refGradients[i][0], gradients[i]);


                // evaluate basis functions
                tFE.localBasis().evaluateFunction(elementQuadPos, values);

                // Evaluate Dirichlet function at quadrature point.
                auto dirichletVal = lf_(elementQuadPos);

                // and vector entries
                for (size_t i=0; i<values.size(); ++i)
                {
                    double factor =quad[pt].weight()*integrationElement*(penalty*values[i] + DGType_*(gradients[i]*outerNormal));
                    localVector[i].axpy(factor, dirichletVal);
                }
            }
        }

        template <class TrialLocalFE, class BoundaryIterator>
        void assembleNeumann(const BoundaryIterator& it, LocalVector& localVector, const TrialLocalFE& tFE)
        {
            using RangeType = typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType;

            localVector = 0.0;

            // geometry of the boundary face
            const auto& segmentGeometry = it->geometry();

            // get quadrature rule
            QuadratureRuleKey tFEquad(it->type(), tFE.localBasis().order());
            auto quadKey = tFEquad.square();

            const auto& quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

            // store values of shape functions
            std::vector<RangeType> values(tFE.localBasis().size());

            const auto& inside = it->inside();
            lf_.bind(inside);

            // loop over quadrature points
            for (size_t pt=0; pt < quad.size(); ++pt)
            {
                // get quadrature point
                const auto& quadPos = quad[pt].position();

                // get integration factor
                const auto integrationElement = segmentGeometry.integrationElement(quadPos);

                // position of the quadrature point within the element
                const auto elementQuadPos = it->geometryInInside().global(quadPos);

                // evaluate basis functions
                tFE.localBasis().evaluateFunction(elementQuadPos, values);

                // Evaluate Neumann function at quadrature point. If it is a grid function use that to speed up the evaluation
                auto neumannVal = lf_(elementQuadPos);

                // and vector entries
                for (size_t i=0; i<values.size(); ++i)
                {
                    double factor =quad[pt].weight()*integrationElement*values[i];
                    localVector[i].axpy(factor, neumannVal);
                }
            }
        }

        void setDGType(double dg)
        {
            if ((dg!=1.0) and (dg!=0.0) and (dg!=-1.0))
                DUNE_THROW(Dune::RangeError, "DG type must be -1, 0 or 1!");
            else
                DGType_ = dg;
        }

        void setPenalty(double penalty) {
            penalty_ = penalty;
        }

    private:
        double penalty_ = 10.0;
        const std::shared_ptr<const Function> bdrFunction_;
        mutable LocalFunction lf_;
        
        // quadrature order
        bool dirichlet;
        bool varyingDegree_;
        double DGType_ = -1.0;
};
}
#endif

