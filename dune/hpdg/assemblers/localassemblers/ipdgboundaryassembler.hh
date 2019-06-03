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
        using ctype = typename GridType::ctype;
        static const int dimworld = GridType::dimensionworld;

        using GlobalCoordinate = typename GridType::template Codim<0>::Geometry::GlobalCoordinate;
        using GridFunction = VirtualGridFunction<GridType, T>;

    public:
        using LocalVector = typename LocalBoundaryAssembler<GridType,T>::LocalVector;

        using Function = typename Dune::VirtualFunction<GlobalCoordinate, T>;

        /** \brief Constructor
         * \param neumann Neumann force function
         * \param order The quadrature order used for numerical integration
         */
        template <class F>
        IPDGBoundaryAssembler(const F& neumann, bool bdrType=true, bool varyingDegree=false) :
            bdrFunction_(Dune::stackobject_to_shared_ptr(neumann)),
            dirichlet(bdrType),
            varyingDegree_(varyingDegree)
        {}

        /** \brief Constructor
         * \param f Neumann force function or Dirichlet data
         * \param order The quadrature order used for numerical integration
         * \param bdrType True means we assemble for Dirichlet data, false means Neumann data
         */
        template <class F>
        IPDGBoundaryAssembler(std::shared_ptr<const F> f, bool bdrType=true)
            : bdrFunction_(f),
              dirichlet(bdrType)
        {}

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
            using FVdim = typename Dune::template FieldVector<ctype,dim>;
            using FV = typename Dune::template FieldVector<ctype,T::dimension>;
            using RangeType = typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType;

            localVector = 0.0;

            double penalty = sigma0;
            if (varyingDegree_) {
                auto degree = tFE.localBasis().order();
                penalty*= degree*degree;
            }
            const auto edgeLength = it->geometry().volume();
            penalty/=edgeLength;


            // geometry of the boundary face
            const typename BoundaryIterator::Intersection::Geometry segmentGeometry = it->geometry();


            // get quadrature rule
            QuadratureRuleKey tFEquad(it->type(), tFE.localBasis().order());
            QuadratureRuleKey quadKey = tFEquad.square();

            const Dune::template QuadratureRule<double, dim-1>& quad = QuadratureRuleCache<double, dim-1>::rule(quadKey);

            // store values of shape functions
            std::vector<RangeType> values(tFE.localBasis().size());
            std::vector<typename TrialLocalFE::Traits::LocalBasisType::Traits::JacobianType> refGradients(tFE.localBasis().size());
            std::vector<Dune::FieldVector<ctype, dimworld>> gradients(tFE.localBasis().size());

            const auto inside = it->inside();

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

                const GridFunction* gf = dynamic_cast<const GridFunction*>(bdrFunction_.get());
                if (gf and gf->isDefinedOn(inside))
                    gf->evaluateLocal(inside, elementQuadPos, dirichletVal);
                else
                    bdrFunction_->evaluate(segmentGeometry.global(quadPos), dirichletVal);

                // and vector entries
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
    private:
        const std::shared_ptr<const Function> bdrFunction_;
        
        // quadrature order
        bool dirichlet;
        bool varyingDegree_;
        double DGType_ = -1.0;
};

#endif

