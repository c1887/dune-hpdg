// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_FUFEM_DUNE_FUNCTIONS_SUBGRID_L2_FUNCTIONAL_ASSEMBLER_HH
#define DUNE_FUFEM_DUNE_FUNCTIONS_SUBGRID_L2_FUNCTIONAL_ASSEMBLER_HH

#include <memory>

#include <dune/common/fvector.hh>

#include <dune/istl/bvector.hh>

#include <dune/fufem/quadraturerules/quadraturerulecache.hh>
#include <dune/fufem/assemblers/localfunctionalassembler.hh>

namespace Dune {
namespace Fufem {

/**  \brief Local assembler for finite element L^2-functionals on the Subgrid, given by Hostgrid-functions
 *
 *  This is needed, e.g. when assembling the right hand side of the spatial problem of a time-discretized time dependent problem with spatial adaptivity.
 *  The solution of the old time step lives on the hostgrid while the rhs is assembled on the NEW subgrid.
 */
template <class G, class TrialLocalFE, class F, class T=Dune::FieldVector<double,1> >
class DuneFunctionsSubgridL2FunctionalAssembler :
  public LocalFunctionalAssembler<G, TrialLocalFE, T>
{
  private:
    using GridType = G;
    using Function = F;
    //using LocalFunction = std::decay_t<decltype(localFunction(std::declval<Function>()))>;
    using LocalFunction = typename Function::LocalFunction;
    static const int dim = GridType::dimension;


  public:
    using Element = typename LocalFunctionalAssembler<GridType,TrialLocalFE, T>::Element;
    using LocalVector = typename LocalFunctionalAssembler<GridType,TrialLocalFE, T>::LocalVector;

    /** \brief constructor
     *
     * Creates a local functional assembler for an L2-functional.
     * It can assemble functionals on the subgrid given by grid
     * functions on the underlying hostgrid exactly.
     *
     * \param f the (hostgrid) function representing the functional
     * \param fQuadKey A QuadratureRuleKey that specifies how to integrate f
     * \param grid the subgrid (!)
     */
    template<class FF>
    DuneFunctionsSubgridL2FunctionalAssembler(FF&& f, const QuadratureRuleKey& fQuadKey, const GridType& grid) :
      f_(Dune::wrap_or_move<const Function>(std::forward<FF>(f))),
      localF_(localFunction(*f_)),
      functionQuadKey_(fQuadKey),
      grid_(grid)
    {}

    void assemble(const Element& element, LocalVector& localVector, const TrialLocalFE& tFE) const
    {
      using LocalBasisRangeType = typename TrialLocalFE::Traits::LocalBasisType::Traits::RangeType;

      // get geometry and store it
      const auto geometry = element.geometry();

      localVector = 0.0;

      const auto baseHostElement = grid_.template getHostEntity<0>(element);

      // store values of shape functions
      std::vector<LocalBasisRangeType> values(tFE.localBasis().size());

      if (baseHostElement.isLeaf())
      {

        localF_.bind(baseHostElement);

        // get quadrature rule
        QuadratureRuleKey tFEquad(tFE);
        QuadratureRuleKey quadKey = tFEquad.product(functionQuadKey_);

        const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

        // loop over quadrature points
        for (size_t pt=0; pt < quad.size(); ++pt)
        {
          // get quadrature point
          const auto& quadPos = quad[pt].position();

          // get integration factor
          const auto integrationElement = geometry.integrationElement(quadPos);

          // evaluate basis functions
          tFE.localBasis().evaluateFunction(quadPos, values);

          // compute values of function
          auto f_pos = localF_(quadPos);

          // and vector entries
          for (size_t i=0; i<values.size(); ++i)
          {
            localVector[i].axpy(values[i]*quad[pt].weight()*integrationElement, (T) f_pos);
          }
        }

      }
      else // corresponding hostgrid element is not in hostgrid leaf
      {
        for (const auto& hostElement : descendantElements(baseHostElement, grid_.getHostGrid().maxLevel()))
        {
          if (hostElement.isLeaf())
          {
            localF_.bind(hostElement);
            const auto hostGeometry = hostElement.geometry();

            // get quadrature rule
            QuadratureRuleKey tFEquad(tFE);
            QuadratureRuleKey quadKey = tFEquad.product(functionQuadKey_);
            const auto& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

            // loop over quadrature points
            for (size_t pt=0; pt < quad.size(); ++pt)
            {
              // get quadrature point
              const auto& quadPos = quad[pt].position();
              const auto quadPosInSubgridElement = geometry.local(hostGeometry.global(quadPos)) ;

              // get integration factor
              const auto integrationElement = hostGeometry.integrationElement(quadPos);

              // evaluate basis functions
              tFE.localBasis().evaluateFunction(quadPosInSubgridElement, values);

              // compute values of function
              auto f_pos = localF_(quadPos);

              // and vector entries
              for (size_t i=0; i<values.size(); ++i)
              {
                localVector[i].axpy(values[i]*quad[pt].weight()*integrationElement, (T) f_pos);
              }
            }
          }
        }
      }
    }

  private:

    std::shared_ptr<const Function> f_;
    mutable LocalFunction localF_;
    const QuadratureRuleKey functionQuadKey_;
    const GridType& grid_;
};

template<class G, class LFE, class F>
auto makeDuneFunctionsSubgridL2FunctionalAssembler(const F& f, QuadratureRuleKey quadKey, const G& grid) {
  return DuneFunctionsSubgridL2FunctionalAssembler<G, LFE, F>(f, quadKey, grid);
}

} // end namespace Fufem
} // end namespace Dune

#endif
