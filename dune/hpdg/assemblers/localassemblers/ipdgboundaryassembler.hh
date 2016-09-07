// -*- tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set ts=8 sw=4 et sts=4:
#ifndef IPDG_BOUNDARY_ASSEMBLER_HH
#define IPDG_BOUNDARY_ASSEMBLER_HH

#include <memory>

#include <dune/common/function.hh>
#include <dune/common/fvector.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/istl/bvector.hh>

#include <dune/fufem/quadraturerules/quadraturerulecache.hh>
#include <dune/fufem/functions/virtualgridfunction.hh>

#include <dune/fufem/assemblers/localboundaryassembler.hh>

/** \brief Assemble right hand side contributions of the IPDG method
 * 
 * \tparam GridType The grid we are assembling for
 * \tparam T Type used for the set of dofs at one node
 */
template <class GridType, class T=Dune::FieldVector<typename GridType::ctype,1> >
class IPDGBoundaryAssembler :
    public LocalBoundaryAssembler<GridType, T>

{
    private:
        static const int dim = GridType::dimension;
        typedef typename GridType::ctype ctype;
        static const int dimworld = GridType::dimensionworld;

        typedef typename GridType::template Codim<0>::Geometry::GlobalCoordinate GlobalCoordinate;
        typedef VirtualGridFunction<GridType, T> GridFunction;

    public:
        typedef typename LocalBoundaryAssembler<GridType,T>::LocalVector LocalVector;

        typedef typename Dune::VirtualFunction<GlobalCoordinate, T> Function;

        /** \brief Constructor
         * \param neumann Neumann force function
         * \param order The quadrature order used for numerical integration
         */
        template <class F>
        IPDGBoundaryAssembler(const F& neumann, int order=2) :
            neumann_(Dune::stackobject_to_shared_ptr(neumann)),
            order_(order)
        {}

        /** \brief Constructor
         * \param neumann Neumann force function
         * \param order The quadrature order used for numerical integration
         */
        template <class F>
        IPDGBoundaryAssembler(std::shared_ptr<const F> neumann, int order=2)
            : neumann_(neumann)
            , order_(order)
        {
            /* Nothing. */
        }

        // TODO:
        template <class TrialLocalFE, class BoundaryIterator>
        void assemble(const BoundaryIterator& it, LocalVector& localVector, const TrialLocalFE& tFE)
        {
            if (dirichlet)
                assembleDirichlet(it, localVector, tFE);
            else
                DUNE_THROW(Dune::NotImplemented, "Neumann data not yet implemented");
//                assembleNeumann();
        }

        template <class TrialLocalFE, class BoundaryIterator>
        void assembleDirichlet(const BoundaryIterator& it, LocalVector& localVector, const TrialLocalFE& tFE)
        {
            typedef typename Dune::template FieldVector<ctype,dim> FVdim;
            typedef typename Dune::template FieldVector<ctype,T::dimension> FV;
            typedef typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType RangeType;

            localVector = 0.0;

            // geometry of the boundary face
            const typename BoundaryIterator::Intersection::Geometry segmentGeometry = it->geometry();


            // get quadrature rule
//            const Dune::QuadratureRule<ctype, dim-1>& quad = QuadratureRuleCache<ctype, dim-1>::rule(segmentGeometry.type(), order_, IsRefinedLocalFiniteElement<TrialLocalFE>::value(tFE) );
//            const Dune::QuadratureRule<ctype, dim-1>& quad = QuadratureRuleCache<ctype, dim-1>::rule(segmentGeometry.type(), order_, IsRefinedLocalFiniteElement<TrialLocalFE>::value(tFE) );
            // get quadrature rule
            QuadratureRuleKey tFEquad(it->type(), tFE.localBasis().order());
            QuadratureRuleKey quadKey = tFEquad.derivative().square();

            const Dune::template QuadratureRule<double, dim-1>& quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

            // store values of shape functions
            std::vector<RangeType> values(tFE.localBasis().size());
            std::vector<typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType> refGradients(tFE.localBasis().size());
            std::vector<Dune::FieldVector<ctype, dimworld>> gradients(tFE.localBasis().size());

            const auto inside = it->inside();

            const auto edgeLength = it->geometry().volume();
            const auto outerNormal = it->centerUnitOuterNormal();


            // loop over quadrature points
            for (size_t pt=0; pt < quad.size(); ++pt)
            {

                // get quadrature point
                const Dune::FieldVector<ctype,dim-1>& quadPos = quad[pt].position();

                // get integration factor
                const ctype integrationElement = segmentGeometry.integrationElement(quadPos);

                // position of the quadrature point within the element
                const FVdim elementQuadPos = it->geometryInInside().global(quadPos);

                // get transposed inverse of Jacobian of transformation
                const auto& invJacobian = inside.geometry().jacobianInverseTransposed(elementQuadPos);

                // Evaluate gradients
                tFE.localBasis().evaluateJacobian(elementQuadPos, refGradients);
                for (size_t i =0; i< gradients.size(); i++)
                    invJacobian.mv(refGradients[i][0], gradients[i]);


                // evaluate basis functions
                tFE.localBasis().evaluateFunction(elementQuadPos, values);


                // Evaluate Dirichlet function at quadrature point. If it is a grid function use that to speed up the evaluation
                FV dirichletVal;

                const GridFunction* gf = dynamic_cast<const GridFunction*>(neumann_.get());
                if (gf and gf->isDefinedOn(inside))
                    gf->evaluateLocal(inside, elementQuadPos, dirichletVal);
                else
                    neumann_->evaluate(segmentGeometry.global(quadPos), dirichletVal);

//                dirichletVal = 1.0; // TODO REMOVE THIS!!11
                // and vector entries
                double penalty = sigma0/edgeLength;
                for (size_t i=0; i<values.size(); ++i)
                {
                    double factor =quad[pt].weight()*integrationElement*(penalty*values[i] + DGType_*(gradients[i]*outerNormal));
                    localVector[i].axpy(factor, dirichletVal);
                }
            }
            return;
        }

        void setDGType(double dg)
        {
            if ((dg!=1.0) and (dg!=0.0) and (dg!=-1.0))
                DUNE_THROW(Dune::RangeError, "DG type must be -1, 0 or 1!");
            else
                DGType_ = dg;
        }

    public:
        double sigma0 = 10.0;
        double dirichlet = true;
    private:
        const std::shared_ptr<const Function> neumann_;
        
        // quadrature order
        const int order_;
        double DGType_ = -1.0;
};

#endif

