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

  };
}
}
#endif
