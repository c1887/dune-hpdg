// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_FUFEM_DUNE_FUNCTIONS_DISCRETIZATION_ERROR_HH
#define DUNE_FUFEM_DUNE_FUNCTIONS_DISCRETIZATION_ERROR_HH

#include <cmath>

#include <dune/geometry/quadraturerules.hh>

#include <dune/fufem/quadraturerules/quadraturerulecache.hh>

namespace Dune {
namespace Fufem {
  template <class GridView>
  class DuneFunctionsDiscretizationError
  {
    enum {dim = GridView::dimension};

    typedef typename GridView::Grid::ctype ctype;

    public:

    /** \brief Compute L2 error between a grid function and an arbitrary function
    */
    template<class F1, class F2>
    static double computeL2Error(const F1& a,
        const F2& b,
        QuadratureRuleKey quadKey)
    {

      // The error to be computed
      double error = 0;

      const auto& gridView = a.entitySet().gridView();

      for (const auto element: elements(gridView)) {

        // Get quadrature formula
        quadKey.setGeometryType(element.type());
        const Dune::QuadratureRule<ctype,dim>& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

        auto localA = localFunction(a);
        localA.bind(element);

        for (size_t i=0; i<quad.size(); i++) {

          // Evaluate function a
          auto aValue = localA(quad[i].position());

          // Evaluate function b.  For now, I assume b can/must be evaluated globally
          auto bValue = b(element.geometry().global(quad[i].position()));

          // integrate error
          error += (aValue - bValue) * (aValue - bValue) * quad[i].weight() * element.geometry().integrationElement(quad[i].position());

        }

      }

      return std::sqrt(error);

    }

    template<class GF, class DF>
    static double computeH1HalfNormError(GF& a, const DF& b, QuadratureRuleKey quadKey) {
      // The error to be computed
      double error = 0;

      //const auto& gridView = a.entitySet().gridView();
      const auto& gridView = a.gridView();

      for (const auto element: elements(gridView)) {

        // Get quadrature formula
        quadKey.setGeometryType(element.type());
        const Dune::QuadratureRule<ctype,dim>& quad = QuadratureRuleCache<double, dim>::rule(quadKey);

        //auto localA = localFunction(a);
        //localA.bind(element);
        a.bind(element);

        for (size_t i=0; i<quad.size(); i++) {

          // Evaluate function a
          //auto aValue = derivative(localA)(quad[i].position());
          auto aValue = a(quad[i].position());

          // Evaluate function b.  For now, I assume b can/must be evaluated globally
          auto bValue = derivative(b)(element.geometry().global(quad[i].position()));

          // integrate error
          aValue -= bValue;
          error += (aValue*aValue) * quad[i].weight() * element.geometry().integrationElement(quad[i].position());

        }

      }
      return std::sqrt(error);
    }

    /** \brief Computes the jump term [[a]] on each edge and sums them up
     *
     * For Dirichlet edges, we compute (a-g) in the L^2 norm on the edge
     */
    template<class F1, class F2>
    static double computeJumpTerm(const F1& a,
                                  const F2& g,
                                  QuadratureRuleKey quadKey)
    {
      // The error to be computed
      double error = 0;

      const auto& gridView = a.entitySet().gridView();

      for (const auto element : elements(gridView)) {
        for (const auto& is : intersections(gridView, element)) {

          // Get quadrature formula
          quadKey.setGeometryType(is.type());
          const Dune::QuadratureRule<ctype, dim - 1>& quad =
            QuadratureRuleCache<double, dim - 1>::rule(quadKey);

          auto localA = localFunction(a);
          localA.bind(element);

          if (not is.neighbor()) {
            for (size_t i = 0; i < quad.size(); i++) {

              // need to lift the quad point to the coordinates of the EDGE
              auto inner_quad_pt =
                is.geometryInInside().global(quad[i].position());

              // Evaluate function a
              auto innerValue = localA(inner_quad_pt);
              auto dirichletValue = g(element.geometry().global(inner_quad_pt));

              // integrate error
              error += (innerValue - dirichletValue) *
                       (innerValue - dirichletValue) * quad[i].weight() *
                       is.geometry().integrationElement(quad[i].position());
            }
          } else {
            const auto& out = is.outside();

            auto out_local = localFunction(a);
            out_local.bind(out);

            for (size_t i = 0; i < quad.size(); i++) {

              // need to lift the quad point to the coordinates of the EDGE
              auto inner_quad_pt =
                is.geometryInInside().global(quad[i].position());
              auto outer_quad_pt =
                is.geometryInOutside().global(quad[i].position());

              // Evaluate function a
              auto innerValue = localA(inner_quad_pt);
              auto outerValue = out_local(outer_quad_pt);

              // integrate error (we put 0.5 here because we visit every edge
              // twice. This is actually wasteful)
              error += 0.5 * (innerValue - outerValue) *
                       (innerValue - outerValue) * quad[i].weight() *
                       is.geometry().integrationElement(quad[i].position());
            }
          }
        }
      }

      return std::sqrt(error);
    }
  };
}
}
#endif
