// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#ifndef DUNE_HPDG_GRIDFUNCTIONS_SIMPLEDERIVATIVE_HH
#define DUNE_HPDG_GRIDFUNCTIONS_SIMPLEDERIVATIVE_HH
#include <vector>
#include <dune/common/fvector.hh>
#include <dune/functions/gridfunctions/gridviewentityset.hh>

namespace Dune {
namespace HPDG {
  /* Allows a the gradient of a function defined by a basis and a coefficient vector(-backend)
   * to be evaluated at points in _local coordinates_ after binding to an element.
   */
  template<typename Basis, typename Vector, typename GradientType>
  class SimpleDerivativeFunction {
    using LV = typename Basis::LocalView;
    using Element = typename LV::Element;
    using Domain = typename Element::Geometry::LocalCoordinate;

    // members:
    const Basis& basis_;
    const Vector& coeffs_;
    LV lv_;
    const Element* e_;

    public:
    SimpleDerivativeFunction(const Basis& b, const Vector& v):
      basis_(b),
      coeffs_(v),
      lv_(basis_.localView()),
      e_(nullptr)
    {}

    const typename Basis::GridView& gridView() const {
      return basis_.gridView();
    }

    void bind(const Element& e) {
      e_ = &e;
      lv_.bind(*e_);
    }

    friend auto localFunction(SimpleDerivativeFunction sdf) {
      return sdf;
    }

    using Range = GradientType;
    using EntitySet = Dune::Functions::GridViewEntitySet<typename Basis::GridView, 0>;

    GradientType operator()(const Domain& x) const { // x is in local (reference element) coordinates!
      const auto& fe = lv_.tree().finiteElement(); // trivial tree only for now
      using FE = std::decay_t<decltype(fe)>;

      using Jac = typename FE::Traits::LocalBasisType::Traits::JacobianType; // FM<1,dim>
      GradientType localGrad; // defined on reference element
      localGrad = 0.0;

      auto derivatives = std::vector<Jac>(fe.size());

      fe.localBasis().evaluateJacobian(x, derivatives);

      for(std::size_t i = 0; i < derivatives.size(); i++) {
        auto&& coeff = coeffs_[lv_.index(i)];
        for (std::size_t d = 0; d < Jac::cols; d++) {
          localGrad[d]+=coeff * derivatives[i][0][d];
        }
      }

      // transform back to element
      auto grad = localGrad;
      grad = 0.0;
      auto J = e_->geometry().jacobianInverseTransposed(x);

      J.mv(localGrad, grad);
      return grad;
    }
  };

  template<class Basis, class Vector>
  auto simpleDerivativeFunction(const Basis& b, const Vector& v) {
    using Grad = Dune::FieldVector<double, Basis::GridView::Grid::dimension>;

    return SimpleDerivativeFunction<Basis, Vector, Grad>(b, v);
  }
}
}
#endif//DUNE_HPDG_GRIDFUNCTIONS_SIMPLEDERIVATIVE_HH
